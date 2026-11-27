"""Migration + upsert tests against a real Postgres (opt-in: set TEST_DATABASE_URL).

WARNING: this test downgrades the target database to an empty schema. Point
TEST_DATABASE_URL at a throwaway database (CI uses a service container), never at the
Neon database you use for development.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
alembic_command = pytest.importorskip("alembic.command")

from alembic.config import Config  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from filinglens.cadence import compute_cadence, merge_filings, parse_filing_block  # noqa: E402
from filinglens.db.models import CadenceSnapshot, Company, Filing, FinancialFact  # noqa: E402
from filinglens.db.session import make_engine, session_scope  # noqa: E402
from filinglens.edgar.tickers import TickerEntry  # noqa: E402
from filinglens.financials import compute_financials  # noqa: E402
from filinglens.pipeline import CompanySnapshot, persist_snapshot  # noqa: E402
from helpers import companyfacts, fact, load_fixture  # noqa: E402

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def migrated(monkeypatch: pytest.MonkeyPatch) -> str:
    import os

    url = os.environ["TEST_DATABASE_URL"]
    monkeypatch.setenv("DATABASE_URL", url)
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "src" / "filinglens" / "db" / "migrations"))
    alembic_command.downgrade(cfg, "base")
    alembic_command.upgrade(cfg, "head")
    return url


def _snapshot() -> CompanySnapshot:
    subs = load_fixture("submissions_example.json")
    filings = merge_filings(
        [
            parse_filing_block(subs["filings"]["recent"]),
            parse_filing_block(load_fixture("submissions_example_page1.json")),
        ]
    )
    cadence = compute_cadence(
        cik="0000000001",
        name="Example Corp",
        fiscal_year_end="1231",
        filings=filings,
        as_of=date(2026, 11, 20),
    )
    facts = companyfacts(
        {
            "Revenues": [
                fact(
                    "2026-04-01",
                    "2026-06-30",
                    150,
                    "0000000001-26-000030",
                    fy=2026,
                    fp="Q2",
                    filed="2026-08-01",
                ),
                fact(
                    "2026-07-01",
                    "2026-09-30",
                    160,
                    "0000000001-26-000040",
                    fy=2026,
                    fp="Q3",
                    filed="2026-11-01",
                ),
            ]
        }
    )
    return CompanySnapshot(
        entry=TickerEntry(ticker="EXMP", cik="0000000001", title="Example Corp"),
        sector="technology",
        submissions=subs,
        filings=[f for f in filings if f.filing_date >= cadence.window_start],
        cadence=cadence,
        financials=compute_financials(facts, cik="0000000001", name="Example Corp"),
    )


def test_persist_is_idempotent(migrated: str) -> None:
    engine = make_engine(migrated)
    snap = _snapshot()
    for _ in range(2):
        with session_scope(engine) as session:
            persist_snapshot(session, snap)

    with session_scope(engine) as session:
        assert session.scalar(select(func.count()).select_from(Company)) == 1
        assert session.scalar(select(func.count()).select_from(Filing)) == len(snap.filings)
        assert session.scalar(select(func.count()).select_from(CadenceSnapshot)) == 4
        rows = session.scalars(select(FinancialFact).order_by(FinancialFact.period_end)).all()
        assert [(r.fiscal_quarter, int(r.value)) for r in rows] == [(2, 150), (3, 160)]
        assert rows[-1].source_accessions == ["0000000001-26-000040"]
        filing = session.get(Filing, "0000000001-26-000040")
        assert filing is not None
        assert filing.url == (
            "https://www.sec.gov/Archives/edgar/data/1/000000000126000040/exmp-20260930.htm"
        )


def test_agent_run_is_logged(migrated: str) -> None:
    from decimal import Decimal

    from filinglens.agent.runner import AgentRunResult
    from filinglens.db.models import AgentRun
    from filinglens.db.repository import insert_agent_run

    run = AgentRunResult(
        scenario="risk_profile",
        input={"ticker": "EXMP"},
        status="ok",
        output={"ticker": "EXMP"},
        model="claude-opus-5-5",
        tokens_in=2000,
        tokens_out=700,
        cost_usd=Decimal("0.0421"),
        latency_ms=9000,
        num_turns=4,
        tool_calls=[{"name": "mcp__filinglens__get_financials", "input": {"ticker": "EXMP"}}],
        session_id="sess-1",
    )
    engine = make_engine(migrated)
    with session_scope(engine) as session:
        first = insert_agent_run(session, run)
        second = insert_agent_run(session, run)  # append-only: a second row, not an update
    assert second > first
    with session_scope(engine) as session:
        row = session.get(AgentRun, first)
        assert row is not None
        assert row.cost_usd == Decimal("0.042100")
        assert row.tool_calls[0]["name"] == "mcp__filinglens__get_financials"
        assert session.scalar(select(func.count()).select_from(AgentRun)) == 2


def test_sql_risk_store_round_trip(migrated: str) -> None:
    """Migration 0003 + SqlRiskStore: save, classify, search (pgvector), diff, replace."""
    from datetime import date as _date

    from filinglens.db.risk_store import SqlRiskStore
    from filinglens.risk.classify import Category, Classification
    from filinglens.risk.diff import DiffRow
    from filinglens.risk.embed import HashingEmbedder
    from filinglens.risk.store import CompanyRef, NewChunk, NewFactor, NewSection

    store = SqlRiskStore(make_engine(migrated))
    emb = HashingEmbedder()
    company = CompanyRef(cik="0000000001", ticker="EXMP", name="Example Corp", sector="tech")

    def section(accession: str, year: int) -> NewSection:
        return NewSection(
            company=company,
            accession=accession,
            filer_cik="0000000001",
            primary_document="k.htm",
            form="10-K",
            filing_date=_date(year + 1, 2, 20),
            report_date=_date(year, 12, 31),
            fiscal_year=year,
            source_url=f"https://www.sec.gov/Archives/edgar/data/1/{accession.replace('-', '')}/k.htm",
            method="item_1a",
            split_method="headings",
            confidence=1.0,
            needs_review=False,
            word_count=5000,
            text="...",
            embedding_model=emb.name,
        )

    texts = ["supply chain disruption in asia", "cyber attacks on our systems"]
    factors = [NewFactor(i + 1, f"Risk {i + 1}.", "Business Risks", t) for i, t in enumerate(texts)]
    chunks = [
        NewChunk(i + 1, i + 1, f"Risk {i + 1}.", "Business Risks", t, 6, v)
        for i, (t, v) in enumerate(zip(texts, emb.embed_passages(texts), strict=True))
    ]
    _, old_factors = store.save_section(section("0000000001-25-000005", 2024), factors, chunks)
    new, new_factors = store.save_section(section("0000000001-26-000005", 2025), factors, chunks)
    assert [s.fiscal_year for s in store.sections("0000000001")] == [2025, 2024]
    assert new.ticker == "EXMP" and len(new_factors) == 2 and new_factors[0].chunk_ids

    f = new_factors[1]
    store.save_classifications(
        {f.id: Classification("F", [Category.CYBER], 4, "Cyber attacks.", f.chunk_ids)},
        {new_factors[0].id: "missing from response"},
        model="claude-opus-5-5",
        agent_run_id=None,
    )
    stored = {x.id: x for x in store.factors(new.id)}
    assert stored[f.id].categories == ["cyber"] and stored[f.id].cited_chunk_ids == f.chunk_ids
    assert stored[new_factors[0].id].status == "failed"

    hits = store.search_chunks(emb.embed_query("cyber attacks"), limit=3)
    assert hits[0].chunk.text == "cyber attacks on our systems"
    assert {h.section.fiscal_year for h in hits} == {2025}  # latest section only
    assert store.chunks_by_id([hits[0].chunk.id])[0].embedding == hits[0].chunk.embedding
    assert store.section_by_id(new.id) == new

    store.save_diffs(
        "0000000001", 2024, 2025, [DiffRow("unchanged", old_factors[0].id, new_factors[0].id, 1.0, 1.0)]
    )
    assert [d.status for d in store.diffs("0000000001")] == ["unchanged"]

    # Re-saving an accession replaces its section; chunks, factors and diffs cascade.
    again, _ = store.save_section(section("0000000001-26-000005", 2025), factors[:1], chunks[:1])
    assert again.id != new.id and len(store.factors(again.id)) == 1
    assert store.diffs("0000000001") == []


def test_eval_report_is_stored(migrated: str) -> None:
    """Migration 0004: one eval_runs row, one eval_results row per metric and per case."""
    from filinglens.db.models import EvalResult, EvalRun
    from filinglens.db.repository import insert_eval_report
    from filinglens.evals.results import CaseResult, EvalReport, MetricResult

    report = EvalReport(
        suite="smoke",
        started_at="2026-11-26T18:44:30+00:00",
        finished_at="2026-11-26T18:44:51+00:00",
        git_sha=None,
        models={"classifier": "", "judge": ""},
        metrics=[
            MetricResult("financial_xbrl_match", 1.0, {"min": 1.0}, True, 20),
            MetricResult("financial_golden_match", None, {"min": 1.0}, None, 0),
        ],
        cases=[
            CaseResult(
                "AAPL-revenue-FY2026Q4", "financial_xbrl_match", 1.0, True, {"value": date(2026, 9, 26)}
            )
        ],
    )
    engine = make_engine(migrated)
    with session_scope(engine) as session:
        run_id = insert_eval_report(session, report)
    with session_scope(engine) as session:
        run = session.get(EvalRun, run_id)
        assert run is not None and run.passed and run.suite == "smoke"
        rows = session.scalars(select(EvalResult).where(EvalResult.eval_run_id == run_id)).all()
        assert sorted(r.kind for r in rows) == ["case", "metric", "metric"]
        skipped = next(r for r in rows if r.metric == "financial_golden_match")
        assert skipped.passed is None and skipped.score is None
        case = next(r for r in rows if r.kind == "case")
        assert case.details["value"] == "2026-09-26"  # dates made JSON-safe


def test_review_queue_and_append_only_audit_log(migrated: str) -> None:
    """Migration 0005: queue, decide (with audit rows), and the audit log refuses changes."""
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    from filinglens.agent.records import AgentRunResult
    from filinglens.db.models import AgentRun, AuditEntry, ReviewItem
    from filinglens.db.repository import ReviewError, decide_review, enqueue_review, insert_agent_run

    engine = make_engine(migrated)
    run = AgentRunResult(
        "risk_profile",
        {"ticker": "AAPL"},
        "ok",
        output={"a": 1},
        guardrails={"blocked": True, "issues": []},
        repairs=1,
    )
    with session_scope(engine) as session:
        run_id = insert_agent_run(session, run)
        review_id = enqueue_review(session, run_id, "[block] policy: investment advice", "block")
    with session_scope(engine) as session:
        stored = session.get(AgentRun, run_id)
        assert (
            stored is not None
            and stored.guardrails == {"blocked": True, "issues": []}
            and stored.repairs == 1
        )
        with pytest.raises(ReviewError, match="reason"):
            decide_review(session, review_id, decision="approve", reviewer="t", reason=" ")
        decide_review(
            session,
            review_id,
            decision="edit",
            reviewer="tvisha",
            reason="removed the advice",
            edited_output={"a": 2},
        )
    with session_scope(engine) as session:
        item = session.get(ReviewItem, review_id)
        assert item is not None and item.status == "edited" and item.edited_output == {"a": 2}
        with pytest.raises(ReviewError, match="already edited"):
            decide_review(session, review_id, decision="reject", reviewer="t", reason="late")
        actions = session.scalars(select(AuditEntry.action).order_by(AuditEntry.id)).all()
        assert actions == ["review.queued", "review.edited"]
    with pytest.raises(DBAPIError, match="append-only"), session_scope(engine) as session:
        session.execute(text("UPDATE audit_log SET actor = 'someone else'"))


def test_heavy_filer_upsert_stays_under_the_parameter_limit(migrated: str) -> None:
    """JPM-sized filing histories exceed PostgreSQL's 65,535 bind parameters in one INSERT."""
    from datetime import timedelta

    from filinglens.cadence import Filing as FilingRow
    from filinglens.db.repository import upsert_company, upsert_filings

    filings = [
        FilingRow(
            accession=f"0000000001-20-{i:06d}",
            form="8-K",
            filing_date=date(2020, 1, 1) + timedelta(days=i % 2000),
            report_date=None,
            primary_document="d.htm",
        )
        for i in range(9000)  # 9,000 x 8 columns = 72,000 parameters
    ]
    engine = make_engine(migrated)
    with session_scope(engine) as session:
        upsert_company(
            session,
            cik="0000000001",
            ticker="BIG",
            name="Big Bank",
            sector="financials",
            sic=None,
            fiscal_year_end="1231",
        )
        assert upsert_filings(session, "0000000001", filings) == 9000
        assert upsert_filings(session, "0000000001", filings) == 9000  # idempotent
    with session_scope(engine) as session:
        assert session.scalar(select(func.count()).select_from(Filing)) == 9000


def test_prices_round_trip(migrated: str) -> None:
    """Migration 0006: upsert is idempotent and load returns the latest days, oldest first."""
    from datetime import timedelta
    from decimal import Decimal

    from filinglens.db.repository import load_prices, upsert_company, upsert_prices
    from filinglens.prices.alphavantage import PriceBar

    bars = [
        PriceBar(
            date(2026, 11, 1) + timedelta(days=i),
            Decimal(1),
            Decimal(2),
            Decimal("0.5"),
            Decimal(100 + i),
            10 * i,
        )
        for i in range(20)
    ]
    engine = make_engine(migrated)
    with session_scope(engine) as session:
        upsert_company(
            session,
            cik="0000000001",
            ticker="EXMP",
            name="Example",
            sector="tech",
            sic=None,
            fiscal_year_end="1231",
        )
        assert upsert_prices(session, "0000000001", bars, "test") == 20
        assert upsert_prices(session, "0000000001", bars, "test") == 20
    with session_scope(engine) as session:
        latest = load_prices(session, "0000000001", 5)
    assert [b.close for b in latest] == [Decimal(115 + i) for i in range(5)]
