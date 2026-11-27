"""HTTP client for SEC EDGAR's documented JSON endpoints.

Every request is rate limited, retried on transient failures, and cached on disk.
Only documented endpoints are used:
  - https://www.sec.gov/files/company_tickers.json
  - https://data.sec.gov/submissions/CIK##########.json (and its paged files)
  - https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  - https://www.sec.gov/Archives/edgar/data/... (filing documents, e.g. a 10-K's primary HTML)
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import httpx

from filinglens.edgar.cache import DiskCache
from filinglens.edgar.ratelimit import TokenBucket
from filinglens.edgar.retry import RetryPolicy
from filinglens.edgar.useragent import validate_user_agent

log = logging.getLogger(__name__)

TICKERS_URL: Final = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL: Final = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE_URL: Final = "https://data.sec.gov/submissions/{name}"
COMPANYFACTS_URL: Final = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVES_PREFIX: Final = "https://www.sec.gov/Archives/edgar/data/"

HOUR: Final = 3600.0
TTL_TICKERS: Final = 7 * 24 * HOUR
TTL_SUBMISSIONS: Final = 12 * HOUR
TTL_SUBMISSIONS_PAGE: Final = 7 * 24 * HOUR  # older pages rarely change
TTL_COMPANYFACTS: Final = 24 * HOUR
TTL_ARCHIVE: Final = 365 * 24 * HOUR  # filed documents are immutable


class SecError(RuntimeError):
    """Base error for SEC access failures."""


class SecNotFoundError(SecError):
    """The entity or document does not exist on EDGAR (e.g. a company that doesn't file)."""


class SecForbiddenError(SecError):
    """SEC returned 403, almost always a missing or malformed User-Agent."""


def format_cik(cik: int | str) -> str:
    """Return the 10-digit, zero-padded CIK that SEC URLs require."""
    digits = str(cik).strip().upper().removeprefix("CIK")
    if not digits.isdigit() or len(digits) > 10:
        raise ValueError(f"invalid CIK: {cik!r}")
    return digits.zfill(10)


class SecClient:
    def __init__(
        self,
        user_agent: str,
        cache: DiskCache,
        *,
        max_requests_per_second: float = 8.0,
        retry: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 30.0,
    ) -> None:
        self._cache = cache
        self._limiter = TokenBucket(max_requests_per_second, sleep=sleep)
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self.request_count = 0
        self._http = httpx.Client(
            headers={
                "User-Agent": validate_user_agent(user_agent),
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SecClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- core -------------------------------------------------------------------------

    def get_json(self, url: str, ttl_seconds: float) -> Any:
        cached = self._cache.get(url, ttl_seconds)
        if cached is not None:
            return cached
        value = self._fetch(url).json()
        self._cache.set(url, value)
        return value

    def get_text(self, url: str, ttl_seconds: float) -> str:
        cached = self._cache.get(url, ttl_seconds)
        if isinstance(cached, str):
            return cached
        value = self._fetch(url).text
        self._cache.set(url, value)
        return value

    def _fetch(self, url: str) -> httpx.Response:
        attempt = 0
        while True:
            attempt += 1
            self._limiter.acquire()
            self.request_count += 1
            try:
                resp = self._http.get(url)
            except httpx.TransportError as exc:
                if attempt >= self._retry.max_attempts:
                    raise SecError(f"network error fetching {url}: {exc}") from exc
                delay = self._retry.delay(attempt)
                log.warning("network error on %s (attempt %d), retrying in %.1fs", url, attempt, delay)
                self._sleep(delay)
                continue

            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                raise SecNotFoundError(f"not found on EDGAR: {url}")
            if resp.status_code == 403:
                raise SecForbiddenError(
                    f"SEC returned 403 for {url}. Check SEC_USER_AGENT (name + email) "
                    "and that you are under the request-rate limit."
                )
            if self._retry.should_retry_status(resp.status_code) and attempt < self._retry.max_attempts:
                delay = self._retry.delay(attempt, resp.headers.get("Retry-After"))
                log.warning(
                    "HTTP %d on %s (attempt %d), retrying in %.1fs",
                    resp.status_code,
                    url,
                    attempt,
                    delay,
                )
                self._sleep(delay)
                continue
            raise SecError(f"HTTP {resp.status_code} fetching {url}")

    # -- endpoints --------------------------------------------------------------------

    def company_tickers(self) -> dict[str, Any]:
        data: dict[str, Any] = self.get_json(TICKERS_URL, TTL_TICKERS)
        return data

    def submissions(self, cik: int | str) -> dict[str, Any]:
        data: dict[str, Any] = self.get_json(SUBMISSIONS_URL.format(cik=format_cik(cik)), TTL_SUBMISSIONS)
        return data

    def submissions_page(self, name: str) -> dict[str, Any]:
        if "/" in name or "\\" in name or not name.endswith(".json"):
            raise ValueError(f"unexpected submissions page name: {name!r}")
        data: dict[str, Any] = self.get_json(SUBMISSIONS_PAGE_URL.format(name=name), TTL_SUBMISSIONS_PAGE)
        return data

    def company_facts(self, cik: int | str) -> dict[str, Any]:
        data: dict[str, Any] = self.get_json(COMPANYFACTS_URL.format(cik=format_cik(cik)), TTL_COMPANYFACTS)
        return data

    def document(self, url: str) -> str:
        """A filing document from EDGAR Archives (HTML/text), e.g. a 10-K's primary document."""
        if not url.startswith(ARCHIVES_PREFIX) or ".." in url or "?" in url:
            raise ValueError(f"not an EDGAR Archives document URL: {url!r}")
        return self.get_text(url, TTL_ARCHIVE)

    def filing_index(self, cik: int | str, accession: str) -> dict[str, Any]:
        """The filing folder's `index.json` directory listing (documented EDGAR directory browsing)."""
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError(f"not an accession number: {accession!r}")
        url = f"{ARCHIVES_PREFIX}{int(format_cik(cik))}/{accession.replace('-', '')}/index.json"
        data: dict[str, Any] = self.get_json(url, TTL_ARCHIVE)
        return data

    def download(self, url: str, dest: Path, chunk_bytes: int = 1 << 20) -> int:
        """Stream a large documented file (SEC bulk ZIPs) to `dest`. Returns bytes written.

        One rate-limited request; written to a temporary file and renamed when complete, so a
        failed download never leaves a truncated archive behind.
        """
        self._limiter.acquire()
        self.request_count += 1
        tmp = dest.with_suffix(dest.suffix + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self._http.stream("GET", url) as resp:
            if resp.status_code == 403:
                raise SecForbiddenError(f"SEC returned 403 for {url}. Check SEC_USER_AGENT.")
            if resp.status_code != 200:
                raise SecError(f"HTTP {resp.status_code} downloading {url}")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_bytes):
                    fh.write(chunk)
                    written += len(chunk)
        tmp.replace(dest)
        return written
