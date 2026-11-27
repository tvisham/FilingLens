"""Live: Item 1A extraction on real 10-Ks (opt-in: RUN_LIVE=1). No LLM calls, no cost.

Covers the layouts that needed special handling: Apple (bold sentence headings), Amazon
(bold-italic titles), ExxonMobil (run-in headings, predecessor CIK), Chevron (unpunctuated
run-ins), AbbVie (heading next to a table), Wells Fargo (risk factors in the annual
report filed next to a cross-reference 10-K).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from filinglens.cadence import years_before
from filinglens.config import load_universe
from filinglens.edgar.cache import DiskCache
from filinglens.edgar.client import SecClient
from filinglens.pipeline import company_filings, load_ticker_index
from filinglens.risk.documents import extract_risk_section, recent_annual_reports
from filinglens.settings import get_settings

pytestmark = pytest.mark.live
UNIVERSE = load_universe(Path(__file__).parents[2] / "config" / "companies.yaml")


@pytest.fixture(scope="module")
def client():  # type: ignore[no-untyped-def]
    s = get_settings()
    with SecClient(s.sec_user_agent, DiskCache(s.cache_dir)) as c:
        yield c


@pytest.mark.parametrize(
    ("ticker", "min_factors", "review"),
    [("AAPL", 20, False), ("AMZN", 15, False), ("XOM", 12, False), ("CVX", 10, False), ("ABBV", 20, False),
     ("WFC", 15, True)],
)  # fmt: skip
def test_latest_two_10ks_extract(client: SecClient, ticker: str, min_factors: int, review: bool) -> None:
    entry = load_ticker_index(client).resolve(ticker)
    _, filings, _ = company_filings(
        client, entry.cik, years_before(date.today(), 3), UNIVERSE.predecessors_for(ticker)
    )
    reports = recent_annual_reports(filings, entry.cik, client.company_facts(entry.cik))
    assert len(reports) == 2 and reports[0].fiscal_year == reports[1].fiscal_year + 1
    for report in reports:
        r = extract_risk_section(client, report)
        print(
            f"\n{ticker} FY{report.fiscal_year} {r.method}/{r.split_method} conf={r.confidence} "
            f"words={r.word_count} factors={len(r.factors)} chunks={len(r.chunks)} "
            f"src={r.source_url.rsplit('/', 1)[-1]}"
        )
        assert r.split_method == "headings"
        assert len(r.factors) >= min_factors
        assert r.needs_review is review
        assert all(c.token_count <= 512 for c in r.chunks)  # bge-small's limit
        assert {c.factor_ordinal for c in r.chunks} == {f.ordinal for f in r.factors}


def test_bge_small_finds_the_ai_risk(client: SecClient) -> None:
    pytest.importorskip("sentence_transformers")
    from filinglens.risk.embed import BgeSmallEmbedder, cosine

    entry = load_ticker_index(client).resolve("AAPL")
    _, filings, _ = company_filings(client, entry.cik, years_before(date.today(), 2))
    report = recent_annual_reports(filings, entry.cik, client.company_facts(entry.cik), count=1)[0]
    chunks = extract_risk_section(client, report).chunks
    embedder = BgeSmallEmbedder()
    vectors = embedder.embed_passages([c.embedding_text for c in chunks])
    assert len(vectors[0]) == 384
    q = embedder.embed_query("risks from artificial intelligence features")
    best = max(range(len(chunks)), key=lambda i: cosine(q, vectors[i]))
    print("\nbest match:", chunks[best].heading)
    assert "artificial intelligence" in (chunks[best].heading or "").lower()
