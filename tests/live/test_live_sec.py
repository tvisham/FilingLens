"""Live checks against SEC EDGAR (opt-in: RUN_LIVE=1, plus SEC_USER_AGENT in .env).

These assert structure and sanity, not exact figures, so they keep passing as new
filings arrive. Hand-verify the printed numbers against the filings once; those
become golden values for the Phase 4 evals.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("httpx")

from filinglens.config import load_universe
from filinglens.edgar.cache import DiskCache
from filinglens.edgar.client import SecClient
from filinglens.financials import NET_INCOME_CONCEPTS, REVENUE_CONCEPTS
from filinglens.formatting import render_cadence, render_financials
from filinglens.pipeline import load_ticker_index, snapshot_company
from filinglens.settings import get_settings

pytestmark = pytest.mark.live
ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
UNIVERSE = load_universe(Path(__file__).parents[2] / "config" / "companies.yaml")


@pytest.fixture(scope="module")
def client():  # type: ignore[no-untyped-def]
    s = get_settings()
    with SecClient(s.sec_user_agent, DiskCache(s.cache_dir)) as c:
        yield c


@pytest.mark.parametrize("ticker", ["AAPL", "JPM", "COST", "WMT", "XOM"])
def test_snapshot_is_sane(client: SecClient, ticker: str) -> None:
    snap = snapshot_company(
        client,
        load_ticker_index(client),
        ticker,
        as_of=date.today(),
        predecessor_ciks=UNIVERSE.predecessors_for(ticker),
    )
    ten_k = next(f for f in snap.cadence.forms if f.form == "10-K")
    ten_q = next(f for f in snap.cadence.forms if f.form == "10-Q")
    assert 2 <= ten_k.count <= 4
    assert 7 <= ten_q.count <= 10

    assert snap.financials is not None
    revenue, net_income = snap.financials.revenue, snap.financials.net_income
    assert revenue.concept in REVENUE_CONCEPTS
    assert net_income.concept in NET_INCOME_CONCEPTS
    for result in (revenue, net_income):
        assert len(result.latest_two) == 2
        older, newer = result.latest_two
        assert older.period_end < newer.period_end
        assert (newer.period_start - older.period_end).days <= 1  # consecutive quarters
        for q in result.latest_two:
            assert q.fiscal_year is not None and q.fiscal_quarter in (1, 2, 3, 4)
            assert all(ACCESSION.match(a) for a in q.source_accessions)
    assert all(q.value > 0 for q in revenue.latest_two)
    print(f"\n===== {ticker} =====\n{render_cadence(snap.cadence)}\n{render_financials(snap.financials)}")
