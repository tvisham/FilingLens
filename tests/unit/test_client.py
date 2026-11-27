"""SecClient and pipeline tests using httpx.MockTransport (no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")

from filinglens.edgar.cache import DiskCache  # noqa: E402
from filinglens.edgar.client import (  # noqa: E402
    SecClient,
    SecError,
    SecForbiddenError,
    SecNotFoundError,
    format_cik,
)
from filinglens.edgar.retry import RetryPolicy  # noqa: E402
from filinglens.pipeline import load_ticker_index, snapshot_company  # noqa: E402
from helpers import companyfacts, fact, load_fixture  # noqa: E402

UA = "FilingLens Test test@example.edu"


def make_client(handler: Any, tmp_path: Path) -> tuple[SecClient, list[float]]:
    sleeps: list[float] = []
    client = SecClient(
        UA,
        DiskCache(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        retry=RetryPolicy(max_attempts=3, base_delay=0.01),
    )
    return client, sleeps


def test_format_cik() -> None:
    assert format_cik(320193) == "0000320193"
    assert format_cik("CIK0000320193") == "0000320193"
    with pytest.raises(ValueError):
        format_cik("abc")


def test_sends_user_agent_and_caches(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    client, _ = make_client(handler, tmp_path)
    assert client.submissions(1) == {"ok": True}
    assert client.submissions("1") == {"ok": True}  # served from cache
    assert len(seen) == 1
    assert seen[0].headers["User-Agent"] == UA
    assert str(seen[0].url) == "https://data.sec.gov/submissions/CIK0000000001.json"


def test_retries_429_using_retry_after(tmp_path: Path) -> None:
    responses = iter([httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200, json=[])])
    client, sleeps = make_client(lambda r: next(responses), tmp_path)
    assert client.company_facts(1) == []
    assert 1.0 in sleeps


def test_gives_up_after_max_attempts(tmp_path: Path) -> None:
    client, _ = make_client(lambda r: httpx.Response(503), tmp_path)
    with pytest.raises(SecError, match="503"):
        client.company_facts(1)
    assert client.request_count == 3


def test_network_errors_are_retried(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"ok": 1})

    client, _ = make_client(handler, tmp_path)
    assert client.company_facts(1) == {"ok": 1}


def test_404_and_403_are_distinct_errors(tmp_path: Path) -> None:
    client, _ = make_client(lambda r: httpx.Response(404), tmp_path)
    with pytest.raises(SecNotFoundError):
        client.company_facts(1)
    client, _ = make_client(lambda r: httpx.Response(403), tmp_path / "b")
    with pytest.raises(SecForbiddenError, match="SEC_USER_AGENT"):
        client.company_facts(1)


def test_rejects_bad_user_agent(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SecClient("no-email", DiskCache(tmp_path))


def test_rejects_suspicious_page_names(tmp_path: Path) -> None:
    client, _ = make_client(lambda r: httpx.Response(200, json={}), tmp_path)
    with pytest.raises(ValueError):
        client.submissions_page("../../etc/passwd.json")


def test_snapshot_company_end_to_end(tmp_path: Path) -> None:
    routes = {
        "https://www.sec.gov/files/company_tickers.json": {
            "0": {"cik_str": 1, "ticker": "EXMP", "title": "Example Corp"}
        },
        "https://data.sec.gov/submissions/CIK0000000001.json": load_fixture("submissions_example.json"),
        "https://data.sec.gov/submissions/CIK0000000001-submissions-001.json": load_fixture(
            "submissions_example_page1.json"
        ),
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json": companyfacts(
            {
                "Revenues": [
                    fact("2026-04-01", "2026-06-30", 150, "a2", fy=2026, fp="Q2", filed="2026-08-01"),
                    fact("2026-07-01", "2026-09-30", 160, "a3", fy=2026, fp="Q3", filed="2026-11-01"),
                ]
            }
        ),
    }
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        if url in routes:
            return httpx.Response(200, json=routes[url])
        return httpx.Response(404)

    client, _ = make_client(handler, tmp_path)
    snap = snapshot_company(client, load_ticker_index(client), "exmp", as_of=date(2026, 11, 20))
    assert "https://data.sec.gov/submissions/CIK0000000001-submissions-002.json" not in requested
    assert next(f for f in snap.cadence.forms if f.form == "10-Q").count == 9
    assert snap.financials is not None
    assert [q.value for q in snap.financials.revenue.latest_two] == [150, 160]
    assert snap.financials.net_income.concept is None
    assert any("no quarterly USD data" in w for w in snap.warnings)
    assert all(f.filing_date >= date(2023, 11, 20) for f in snap.filings)


def _submissions(name: str, rows: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    """Columnar submissions JSON from (accession, form, filingDate, reportDate) rows."""
    return {
        "name": name,
        "fiscalYearEnd": "1231",
        "filings": {
            "recent": {
                "accessionNumber": [r[0] for r in rows],
                "form": [r[1] for r in rows],
                "filingDate": [r[2] for r in rows],
                "reportDate": [r[3] for r in rows],
                "primaryDocument": ["doc.htm" for _ in rows],
            },
            "files": [],
        },
    }


def test_snapshot_merges_predecessor_filings(tmp_path: Path) -> None:
    """Holding-company reorg: the ticker maps to a new CIK; 10-K history is under the old one."""
    shared = "0000000001-26-000093"  # filed by the old registrant, listed under both CIKs
    routes = {
        "https://www.sec.gov/files/company_tickers.json": {
            "0": {"cik_str": 2, "ticker": "EXMP", "title": "Example Holdings"}
        },
        "https://data.sec.gov/submissions/CIK0000000002.json": _submissions(
            "Example Holdings",
            [
                ("0000000002-26-000020", "10-Q", "2026-11-02", "2026-09-30"),
                (shared, "10-Q", "2026-08-03", "2026-06-30"),
                ("0000000002-26-000001", "8-K12B", "2026-07-01", "2026-07-01"),
            ],
        ),
        "https://data.sec.gov/submissions/CIK0000000001.json": _submissions(
            "Example Corp",
            [
                (shared, "10-Q", "2026-08-03", "2026-06-30"),
                ("0000000001-26-000050", "10-Q", "2026-05-01", "2026-03-31"),
                ("0000000001-26-000010", "10-K", "2026-02-20", "2025-12-31"),
                ("0000000001-25-000010", "10-K", "2025-02-20", "2024-12-31"),
                ("0000000001-24-000010", "10-K", "2024-02-20", "2023-12-31"),
            ],
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        return httpx.Response(200, json=routes[url]) if url in routes else httpx.Response(404)

    client, _ = make_client(handler, tmp_path)
    snap = snapshot_company(
        client,
        load_ticker_index(client),
        "EXMP",
        as_of=date(2026, 11, 24),
        predecessor_ciks=["0000000001", "0000000003"],
    )
    forms = {f.form: f.count for f in snap.cadence.forms}
    assert forms["10-K"] == 3
    assert forms["10-Q"] == 3  # the shared accession is counted once
    assert snap.cadence.cik == "0000000002"
    assert snap.cadence.name == "Example Holdings"
    assert snap.cadence.fiscal_periods[-1].ten_q_count == 3  # FY2026 complete across both CIKs
    filer = {f.accession: f.filer_cik for f in snap.filings}
    assert filer[shared] == "0000000002"  # current registrant's listing wins
    assert filer["0000000001-26-000010"] == "0000000001"
    assert any("0000000003" in w for w in snap.warnings)  # missing predecessor is reported

    repository = pytest.importorskip("filinglens.db.repository")
    old_k = next(f for f in snap.filings if f.accession == "0000000001-26-000010")
    assert repository.filing_url(old_k.filer_cik or "2", old_k.accession, old_k.primary_document) == (
        "https://www.sec.gov/Archives/edgar/data/1/000000000126000010/doc.htm"
    )


def test_snapshot_company_without_xbrl(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 1, "ticker": "EXMP", "title": "X"}})
        if "submissions" in url and url.endswith("CIK0000000001.json"):
            return httpx.Response(200, json=load_fixture("submissions_example.json"))
        if url.endswith("submissions-001.json"):
            return httpx.Response(200, json=load_fixture("submissions_example_page1.json"))
        return httpx.Response(404)

    client, _ = make_client(handler, tmp_path)
    snap = snapshot_company(client, load_ticker_index(client), "EXMP", as_of=date(2026, 11, 20))
    assert snap.financials is None
    assert "no XBRL" in snap.warnings[0]
