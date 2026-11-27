"""Stock prices: Alpha Vantage parsing/refusals/caching, share counts, the tool, the guardrail."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from filinglens.agent.records import AgentRunResult
from filinglens.edgar.cache import DiskCache
from filinglens.guardrails.checks import check_numbers
from filinglens.guardrails.review import guard
from filinglens.mcp_server import tools
from filinglens.prices.alphavantage import (
    AlphaVantageClient,
    PriceBar,
    PriceSourceError,
    latest_share_count,
    market_cap,
    parse_daily,
    provider_symbol,
)
from helpers import mock_sec_client

# Shape verified against Alpha Vantage's documented demo response (TIME_SERIES_DAILY, IBM).
DAILY: dict[str, Any] = {
    "Meta Data": {"1. Information": "Daily Prices (open, high, low, close) and Volumes", "2. Symbol": "EXMP"},
    "Time Series (Daily)": {
        "2026-11-25": {
            "1. open": "233.5100",
            "2. high": "246.2400",
            "3. low": "232.1700",
            "4. close": "243.4700",
            "5. volume": "5231889",
        },
        "2026-11-24": {
            "1. open": "230.0000",
            "2. high": "234.0000",
            "3. low": "229.5000",
            "4. close": "233.1000",
            "5. volume": "4100000",
        },
    },
}
REFUSAL = {"Information": "We have detected your API key as ... 25 requests per day ..."}


def test_parse_daily_orders_and_types() -> None:
    bars = parse_daily(DAILY)
    assert [b.day for b in bars] == [date(2026, 11, 24), date(2026, 11, 25)]
    assert bars[-1].close == Decimal("243.4700") and bars[-1].volume == 5231889
    for refusal in (REFUSAL, {"Note": "slow down"}, {"Error Message": "Invalid API call"}, {}):
        with pytest.raises(PriceSourceError):
            parse_daily(refusal)
    assert provider_symbol("brk-b") == "BRK.B"


def _client(tmp_path: Path, responses: list[dict[str, Any]], seen: list[httpx.Request]) -> AlphaVantageClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=responses.pop(0))

    return AlphaVantageClient(
        "SECRET-KEY", DiskCache(tmp_path), transport=httpx.MockTransport(handler), sleep=lambda _s: None
    )


def test_client_caches_without_the_key_and_never_caches_refusals(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    with _client(tmp_path, [REFUSAL, DAILY], seen) as av:
        with pytest.raises(PriceSourceError, match="25 requests"):
            av.daily("EXMP")
        assert len(av.daily("exmp")) == 2  # refusal was not cached: a second request is made
        assert len(av.daily("EXMP")) == 2  # now cached
        assert av.request_count == 2
    assert (
        seen[0].url.params["function"] == "TIME_SERIES_DAILY"
        and seen[0].url.params["outputsize"] == "compact"
    )
    cached = [json.loads(p.read_text(encoding="utf-8")) for p in tmp_path.rglob("*.json")]
    assert cached and all("SECRET-KEY" not in json.dumps(c) for c in cached)


def test_empty_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alphavantage"):
        AlphaVantageClient(" ", DiskCache(tmp_path))


def _shares(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for qualified, values in rows.items():
        taxonomy, concept = qualified.split(":")
        facts.setdefault(taxonomy, {})[concept] = {"units": {"shares": values}}
    return {"facts": facts}


def test_share_count_sums_classes_from_the_latest_filing() -> None:
    facts = _shares(
        {
            "dei:EntityCommonStockSharesOutstanding": [
                {"end": "2026-07-20", "val": 100, "accn": "old", "filed": "2026-07-30"},
                {"end": "2026-10-20", "val": 600, "accn": "new", "filed": "2026-10-30", "frame": "A"},
                {"end": "2026-10-20", "val": 400, "accn": "new", "filed": "2026-10-30", "frame": "B"},
            ]
        }
    )
    count = latest_share_count(facts)
    assert count is not None and (count.shares, count.accession) == (1000, "new")
    assert market_cap(Decimal("243.47"), count) == Decimal(243470)
    # Alphabet-style: no cover-page concept, balance-sheet total instead.
    fallback = latest_share_count(
        _shares({"us-gaap:CommonStockSharesOutstanding": [{"end": "2026-09-30", "val": 12294, "accn": "q3"}]})
    )
    assert fallback is not None and fallback.concept == "us-gaap:CommonStockSharesOutstanding"
    assert latest_share_count({"facts": {}}) is None


def test_price_tool(tmp_path: Path) -> None:
    facts = _shares(
        {"dei:EntityCommonStockSharesOutstanding": [{"end": "2026-10-20", "val": 1000, "accn": "k"}]}
    )
    client = mock_sec_client(
        tmp_path,
        {
            "https://www.sec.gov/files/company_tickers.json": {
                "0": {"cik_str": 1, "ticker": "EXMP", "title": "X"}
            },
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json": facts,
        },
    )
    bars = parse_daily(DAILY)
    calls: list[tuple[str, int]] = []

    def loader(cik: str, days: int) -> list[PriceBar]:
        calls.append((cik, days))
        return bars

    res = tools.get_price_history(tools.ToolContext(client=client, price_loader=loader), "exmp", days=500)
    assert calls == [("0000000001", 100)]  # capped
    assert res.status == "available" and res.last_close_usd == Decimal("243.4700")
    assert res.market_cap_usd == Decimal(243470) and "recommendations" in res.note
    empty = tools.get_price_history(tools.ToolContext(client=client, price_loader=lambda c, d: []), "EXMP")
    assert empty.status == "not_available" and "prices fetch EXMP" in empty.message
    assert tools.get_price_history(tools.ToolContext(client=client), "EXMP").status == "not_available"


def test_guardrail_accepts_prices_a_tool_returned_but_not_date_fragments() -> None:
    tool_text = str(
        {"closes": [{"day": "2026-11-25", "close": Decimal("243.4700")}], "market_cap_usd": "3538000000000"}
    )
    assert check_numbers({"obs": ["It closed at $243.47 on Nov 25."]}, [], tool_text) == []
    assert check_numbers({"obs": ["Market cap is about $3.54 trillion."]}, [], tool_text) == []
    (issue,) = check_numbers({"obs": ["A $25 fee applies."]}, [], tool_text)  # 25 only appears in a date
    assert issue.severity == "review"
    run = AgentRunResult(
        "question",
        {},
        "ok",
        output={"answer": "It is a strong buy at $243.47.", "disclaimer": ""},
        tool_payloads=[{"close": "243.4700"}],
    )
    assert guard(run, []).blocked  # prices never make advice acceptable


def test_stale_share_count_gives_no_market_cap() -> None:
    """Visa: the only undimensioned cover-page count is from 2010."""
    facts = _shares(
        {"dei:EntityCommonStockSharesOutstanding": [{"end": "2010-01-27", "val": 469280842, "accn": "q"}]}
    )
    assert latest_share_count(facts, not_before=date(2025, 6, 1)) is None
    assert latest_share_count(facts) is not None  # without a cutoff the stale value is still visible
