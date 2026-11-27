"""FilingLens tool functions over a mocked SEC client (no network, no MCP)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from filinglens.config import Universe
from filinglens.mcp_server import tools
from helpers import companyfacts, fact, mock_sec_client, submissions

TODAY = date(2026, 11, 24)
TICKERS = "https://www.sec.gov/files/company_tickers.json"
SUBS = "https://data.sec.gov/submissions/CIK0000000001.json"
FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"


def routes() -> dict[str, Any]:
    return {
        TICKERS: {
            "0": {"cik_str": 1, "ticker": "EXMP", "title": "Example Corp"},
            "1": {"cik_str": 9, "ticker": "NOXB", "title": "No XBRL Inc"},
        },
        SUBS: submissions(
            "Example Corp",
            [
                ("0000000001-26-000040", "8-K", "2026-11-10", "2026-11-06", "5.02", ""),
                ("0000000001-26-000030", "10-Q", "2026-11-01", "2026-09-30", "", "FORM 10-Q"),
                ("0000000001-26-000025", "8-K", "2026-10-28", "2026-10-28", "2.02,9.01", "EARNINGS RELEASE"),
                ("0000000001-26-000020", "10-Q", "2026-08-01", "2026-06-30", "", "FORM 10-Q"),
                ("0000000001-26-000010", "10-Q", "2026-05-01", "2026-03-31", "", "FORM 10-Q"),
                ("0000000001-26-000005", "10-K", "2026-02-20", "2025-12-31", "", "ANNUAL REPORT"),
                ("0000000001-25-000050", "8-K", "2025-03-03", "2025-03-01", "5.02,9.01", ""),
                ("0000000001-19-000001", "8-K", "2019-05-01", "2019-05-01", "5.02", ""),
            ],
            sic="1234",
            sicDescription="Widgets",
            exchanges=["NYSE", ""],
            formerNames=[{"name": "Example Holdings"}],
        ),
        FACTS: companyfacts(
            {
                "Revenues": [
                    fact(
                        "2026-01-01",
                        "2026-03-31",
                        140,
                        "0000000001-26-000010",
                        fy=2026,
                        fp="Q1",
                        filed="2026-05-01",
                    ),
                    fact(
                        "2026-04-01",
                        "2026-06-30",
                        150,
                        "0000000001-26-000020",
                        fy=2026,
                        fp="Q2",
                        filed="2026-08-01",
                    ),
                    fact(
                        "2026-07-01",
                        "2026-09-30",
                        160.5,
                        "0000000001-26-000030",
                        fy=2026,
                        fp="Q3",
                        filed="2026-11-01",
                    ),
                ]
            }
        ),
    }


@pytest.fixture()
def ctx(tmp_path: Path) -> tools.ToolContext:
    universe = Universe.model_validate({"sectors": {"industrials": ["EXMP"]}})
    return tools.ToolContext(
        client=mock_sec_client(tmp_path, routes()), universe=universe, today=lambda: TODAY
    )


def test_lookup_company(ctx: tools.ToolContext) -> None:
    info = tools.lookup_company(ctx, "exmp")
    assert (info.ticker, info.cik, info.name) == ("EXMP", "0000000001", "Example Corp")
    assert info.in_universe and info.sector == "industrials"
    assert info.sic_description == "Widgets"
    assert info.exchanges == ["NYSE"]  # blank exchange dropped
    assert info.former_names == ["Example Holdings"]


def test_unknown_ticker_is_a_tool_input_error(ctx: tools.ToolContext) -> None:
    with pytest.raises(tools.ToolInputError, match="unknown ticker"):
        tools.lookup_company(ctx, "PRIVATECO")


def test_filing_index_counts_filters_and_links(ctx: tools.ToolContext) -> None:
    idx = tools.get_filing_index(ctx, "EXMP", forms=["10-q"], limit=2)
    counts = {c.form: c.count for c in idx.counts}
    assert counts == {"10-K": 1, "10-Q": 3, "8-K": 3, "DEF 14A": 0}  # 2019 8-K is outside the window
    assert idx.total_matching == 3
    assert [f.accession for f in idx.filings] == ["0000000001-26-000030", "0000000001-26-000020"]
    assert idx.filings[0].url == "https://www.sec.gov/Archives/edgar/data/1/000000000126000030/doc1.htm"
    assert idx.fiscal_years[-1].in_progress and idx.fiscal_years[-1].ten_q_count == 3


def test_filing_index_clamps_limits(ctx: tools.ToolContext) -> None:
    idx = tools.get_filing_index(ctx, "EXMP", years=99, limit=0)
    assert len(idx.filings) == 1
    assert idx.window_start == date(2016, 11, 24)  # clamped to MAX_YEARS


def test_financials_are_exact_and_sourced(ctx: tools.ToolContext) -> None:
    fin = tools.get_financials(ctx, "EXMP", quarters=2)
    assert fin.revenue.concept == "Revenues"
    assert [q.value_usd for q in fin.revenue.quarters] == [Decimal(150), Decimal("160.5")]
    assert fin.revenue.quarters[-1].accession == "0000000001-26-000030"
    assert (fin.revenue.quarters[-1].fiscal_year, fin.revenue.quarters[-1].fiscal_quarter) == (2026, 3)
    assert fin.net_income.concept is None and fin.net_income.warnings
    # Decimal survives the JSON round trip exactly.
    assert fin.model_dump(mode="json")["revenue"]["quarters"][-1]["value_usd"] == "160.5"


def test_financials_without_xbrl(ctx: tools.ToolContext) -> None:
    with pytest.raises(tools.ToolInputError, match="no XBRL"):
        tools.get_financials(ctx, "NOXB")


def test_risk_factors_need_a_store(ctx: tools.ToolContext) -> None:
    with pytest.raises(tools.ToolInputError, match="not configured"):
        tools.get_risk_factors(ctx, "EXMP")  # full coverage in test_risk_pipeline.py


def test_search_by_8k_item(ctx: tools.ToolContext) -> None:
    res = tools.search_filings(ctx, "EXMP", items=["5.02"])
    assert [f.accession for f in res.filings] == ["0000000001-26-000040", "0000000001-25-000050"]
    assert "not the documents' full text" in res.search_scope


def test_search_by_keywords_and_dates(ctx: tools.ToolContext) -> None:
    res = tools.search_filings(ctx, "EXMP", query="earnings release")
    assert [f.accession for f in res.filings] == ["0000000001-26-000025"]
    res = tools.search_filings(
        ctx, "EXMP", forms=["8-K"], start_date=date(2019, 1, 1), end_date=date(2019, 12, 31)
    )
    assert [f.accession for f in res.filings] == ["0000000001-19-000001"]
    with pytest.raises(tools.ToolInputError, match="after end_date"):
        tools.search_filings(ctx, "EXMP", start_date=date(2026, 1, 2), end_date=date(2026, 1, 1))
