"""End-of-day stock prices from Alpha Vantage (free API key; https://www.alphavantage.co).

Free keys get `TIME_SERIES_DAILY` with `outputsize=compact` (the latest 100 trading days)
and 25 requests per day. Responses are cached on disk for 12 hours and requests are
spaced out (5 per minute), so refreshing 25 companies fits in one day's quota.

Alpha Vantage reports refusals (rate limit, bad key) as HTTP 200 with an `Information`,
`Note` or `Error Message` field instead of data; those raise `PriceSourceError`. The API key
is never part of a cache key or a log line.

Prices are factual context next to the filings. FilingLens does not turn them into
valuations, ratings, recommendations or price targets.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import httpx

from filinglens.edgar.cache import DiskCache
from filinglens.edgar.ratelimit import TokenBucket

API_URL = "https://www.alphavantage.co/query"
SOURCE = "Alpha Vantage TIME_SERIES_DAILY (end of day, not adjusted for splits or dividends)"
TTL_SECONDS = 12 * 3600
REQUESTS_PER_MINUTE = 5
_REFUSAL_KEYS = ("Information", "Note", "Error Message")


class PriceSourceError(RuntimeError):
    """The provider refused or failed (rate limit, invalid key, unknown symbol)."""


@dataclass(frozen=True)
class PriceBar:
    day: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


def provider_symbol(ticker: str) -> str:
    """SEC tickers use "-" for share classes (BRK-B); Alpha Vantage uses "." (BRK.B)."""
    return ticker.strip().upper().replace("-", ".")


def parse_daily(payload: Mapping[str, Any]) -> list[PriceBar]:
    for key in _REFUSAL_KEYS:
        if key in payload:
            raise PriceSourceError(f"Alpha Vantage: {payload[key]}")
    series = payload.get("Time Series (Daily)")
    if not isinstance(series, Mapping) or not series:
        raise PriceSourceError("Alpha Vantage returned no daily price series")
    bars = [
        PriceBar(
            day=date.fromisoformat(day),
            open=Decimal(v["1. open"]),
            high=Decimal(v["2. high"]),
            low=Decimal(v["3. low"]),
            close=Decimal(v["4. close"]),
            volume=int(v["5. volume"]),
        )
        for day, v in series.items()
    ]
    return sorted(bars, key=lambda b: b.day)


class AlphaVantageClient:
    def __init__(
        self,
        api_key: str,
        cache: DiskCache,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        requests_per_minute: float = REQUESTS_PER_MINUTE,
    ) -> None:
        if not api_key.strip():
            raise ValueError(
                "ALPHAVANTAGE_API_KEY is empty: get a free key at https://www.alphavantage.co/support/#api-key"
            )
        self._key = api_key.strip()
        self._cache = cache
        self._limiter = TokenBucket(requests_per_minute / 60, capacity=1, sleep=sleep)
        self._http = httpx.Client(transport=transport, timeout=30, headers={"User-Agent": "FilingLens/0.1"})
        self.request_count = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> AlphaVantageClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def daily(self, ticker: str) -> list[PriceBar]:
        """The latest ~100 trading days for `ticker`, oldest first."""
        symbol = provider_symbol(ticker)
        cache_key = f"alphavantage:TIME_SERIES_DAILY:compact:{symbol}"  # no API key in the key
        cached = self._cache.get(cache_key, TTL_SECONDS)
        if cached is not None:
            return parse_daily(cached)
        self._limiter.acquire()
        self.request_count += 1
        try:
            resp = self._http.get(
                API_URL,
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "outputsize": "compact",
                    "apikey": self._key,
                },
            )
        except httpx.TransportError as exc:
            raise PriceSourceError(
                f"network error fetching prices for {symbol}: {type(exc).__name__}"
            ) from None
        if resp.status_code != 200:
            raise PriceSourceError(f"Alpha Vantage HTTP {resp.status_code} for {symbol}")
        payload = resp.json()
        bars = parse_daily(payload)  # raises before caching a refusal
        self._cache.set(cache_key, payload)
        return bars


# -- shares outstanding and market capitalisation ------------------------------------------


@dataclass(frozen=True)
class ShareCount:
    shares: int
    as_of: date
    concept: str
    accession: str


MAX_SHARE_COUNT_AGE_DAYS = 550  # ~18 months: older counts don't describe today's company


def latest_share_count(
    companyfacts: Mapping[str, Any], *, not_before: date | None = None
) -> ShareCount | None:
    """Shares outstanding from the 10-K/10-Q cover page (`dei:EntityCommonStockSharesOutstanding`,
    summed across share classes in the latest filing), else the balance sheet's
    `us-gaap:CommonStockSharesOutstanding` (Alphabet reports only the latter).

    Counts dated before `not_before` are ignored: Visa's last undimensioned cover-page count
    is from 2010 (its class A/B/C counts are dimensional, which companyfacts leaves out), and a
    market cap from it would be badly wrong. No count is better than a stale one."""
    facts = companyfacts.get("facts", {})
    for taxonomy, concept in (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ):
        rows = facts.get(taxonomy, {}).get(concept, {}).get("units", {}).get("shares", [])
        rows = [r for r in rows if r.get("end") and r.get("val") is not None and r.get("accn")]
        if not_before is not None:
            rows = [r for r in rows if date.fromisoformat(r["end"]) >= not_before]
        if not rows:
            continue
        latest = max(rows, key=lambda r: (r["end"], r.get("filed", "")))
        same = [r for r in rows if r["end"] == latest["end"] and r["accn"] == latest["accn"]]
        total = sum(int(r["val"]) for r in {(r["val"], r.get("frame")): r for r in same}.values())
        return ShareCount(total, date.fromisoformat(latest["end"]), f"{taxonomy}:{concept}", latest["accn"])
    return None


def market_cap(close: Decimal, shares: ShareCount) -> Decimal:
    return (close * shares.shares).quantize(Decimal(1))
