"""Idempotent upserts for Phase 1 data (Postgres `INSERT ... ON CONFLICT`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from filinglens.cadence import CadenceReport
from filinglens.cadence import Filing as FilingRow
from filinglens.cadence import filing_url as filing_url  # re-export; DB-free code imports it from cadence
from filinglens.db.models import AgentRun, CadenceSnapshot, Company, Filing, FinancialFact
from filinglens.financials import MetricResult

if TYPE_CHECKING:
    from filinglens.agent.records import AgentRunResult
    from filinglens.evals.results import EvalReport
    from filinglens.prices.alphavantage import PriceBar


def upsert_company(
    session: Session,
    *,
    cik: str,
    ticker: str,
    name: str,
    sector: str,
    sic: str | None,
    fiscal_year_end: str | None,
) -> None:
    stmt = insert(Company).values(
        cik=cik, ticker=ticker, name=name, sector=sector, sic=sic, fiscal_year_end=fiscal_year_end
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Company.cik],
        set_={
            "ticker": stmt.excluded.ticker,
            "name": stmt.excluded.name,
            "sector": stmt.excluded.sector,
            "sic": stmt.excluded.sic,
            "fiscal_year_end": stmt.excluded.fiscal_year_end,
            "updated_at": func.now(),
        },
    )
    session.execute(stmt)


# PostgreSQL's wire protocol allows at most 65,535 bound parameters per statement; heavy
# filers (JPM: ~3,000 filings x 8 columns) exceed it in one multi-row INSERT.
MAX_ROWS_PER_STATEMENT = 1000


def _batches[T](rows: list[T]) -> list[list[T]]:
    return [rows[i : i + MAX_ROWS_PER_STATEMENT] for i in range(0, len(rows), MAX_ROWS_PER_STATEMENT)]


def upsert_filings(session: Session, cik: str, filings: list[FilingRow]) -> int:
    if not filings:
        return 0
    rows = [
        {
            "accession": f.accession,
            "cik": cik,
            "form": f.form,
            "is_amendment": f.is_amendment,
            "filing_date": f.filing_date,
            "report_date": f.report_date,
            "primary_document": f.primary_document,
            # Archive paths are per filer: a predecessor's filing lives under its own CIK.
            "url": filing_url(f.filer_cik or cik, f.accession, f.primary_document),
        }
        for f in filings
    ]
    for batch in _batches(rows):
        stmt = insert(Filing).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Filing.accession],
            set_={c: getattr(stmt.excluded, c) for c in ("form", "report_date", "primary_document", "url")},
        )
        session.execute(stmt)
    return len(rows)


def upsert_metric(session: Session, cik: str, result: MetricResult) -> int:
    if not result.quarters:
        return 0
    rows = [
        {
            "cik": cik,
            "metric": result.metric.value,
            "concept": q.concept,
            "period_start": q.period_start,
            "period_end": q.period_end,
            "duration_days": q.duration_days,
            "period_type": q.period_type.value,
            "fiscal_year": q.fiscal_year,
            "fiscal_quarter": q.fiscal_quarter,
            "value": q.value,
            "unit": q.unit,
            "accession": q.accession,
            "form": q.form,
            "filed": q.filed,
            "derived": q.derived,
            "derivation": q.derivation,
            "source_accessions": q.source_accessions,
        }
        for q in result.quarters
    ]
    stmt = insert(FinancialFact).values(rows)
    updatable = [k for k in rows[0] if k not in ("cik", "metric", "period_start", "period_end")]
    stmt = stmt.on_conflict_do_update(
        constraint="uq_financial_period",
        set_={k: getattr(stmt.excluded, k) for k in updatable},
    )
    session.execute(stmt)
    return len(rows)


def upsert_cadence(session: Session, report: CadenceReport) -> int:
    rows = [
        {
            "cik": report.cik,
            "window_start": report.window_start,
            "window_end": report.window_end,
            "form": f.form,
            "count": f.count,
            "amendments": f.amendments,
            "dates": f.dates,
            "median_days_after_period_end": f.median_days_after_period_end,
        }
        for f in report.forms
    ]
    stmt = insert(CadenceSnapshot).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_cadence_snapshot",
        set_={
            k: getattr(stmt.excluded, k)
            for k in ("window_start", "count", "amendments", "dates", "median_days_after_period_end")
        }
        | {"computed_at": func.now()},
    )
    session.execute(stmt)
    return len(rows)


def insert_agent_run(session: Session, run: AgentRunResult) -> int:
    """Append one agent run (never updated: the log is an audit trail). Returns its id."""
    row = AgentRun(
        scenario=run.scenario,
        input=run.input,
        output=run.output,
        status=run.status,
        error=run.error,
        model=run.model,
        tokens_in=run.tokens_in,
        tokens_out=run.tokens_out,
        cost_usd=run.cost_usd,
        latency_ms=run.latency_ms,
        num_turns=run.num_turns,
        tool_calls=run.tool_calls,
        session_id=run.session_id,
        guardrails=run.guardrails,
        repairs=run.repairs,
    )
    session.add(row)
    session.flush()
    return row.id


def insert_eval_report(session: Session, report: EvalReport) -> int:
    """One eval_runs row plus an eval_results row per metric and per case."""
    import json
    from datetime import datetime
    from decimal import Decimal

    from filinglens.db.models import EvalResult, EvalRun

    run = EvalRun(
        suite=report.suite,
        git_sha=report.git_sha,
        models=report.models,
        started_at=datetime.fromisoformat(report.started_at),
        finished_at=datetime.fromisoformat(report.finished_at),
        passed=report.passed,
        cost_usd=Decimal(str(report.cost_usd)),
    )
    session.add(run)
    session.flush()

    def clean(d: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = json.loads(json.dumps(d, default=str))  # JSONB-safe (dates, Decimals)
        return out

    rows = [
        EvalResult(
            eval_run_id=run.id,
            kind="metric",
            metric=m.metric,
            case_id=None,
            score=m.value,
            threshold=m.threshold,
            passed=m.passed,
            details=clean(m.details | {"n_cases": m.n_cases}),
        )
        for m in report.metrics
    ] + [
        EvalResult(
            eval_run_id=run.id,
            kind="case",
            metric=c.metric,
            case_id=c.case_id[:200],
            score=c.score,
            threshold=None,
            passed=c.passed,
            details=clean(c.details),
        )
        for c in report.cases
    ]
    session.add_all(rows)
    return run.id


# -- Phase 5: review queue and audit log --------------------------------------------------


def audit(
    session: Session,
    actor: str,
    action: str,
    entity: str,
    entity_id: int | None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append one audit_log row (the table refuses updates and deletes)."""
    import json

    from filinglens.db.models import AuditEntry

    clean = json.loads(json.dumps(payload or {}, default=str))
    session.add(AuditEntry(actor=actor, action=action, entity=entity, entity_id=entity_id, payload=clean))


def enqueue_review(session: Session, agent_run_id: int, reason: str, severity: str) -> int:
    from filinglens.db.models import ReviewItem

    item = ReviewItem(agent_run_id=agent_run_id, reason=reason, severity=severity, status="pending")
    session.add(item)
    session.flush()
    audit(
        session,
        "system",
        "review.queued",
        "review_queue",
        item.id,
        {"agent_run_id": agent_run_id, "reason": reason},
    )
    return item.id


REVIEW_DECISIONS = {"approve": "approved", "edit": "edited", "reject": "rejected"}


class ReviewError(ValueError):
    pass


def decide_review(
    session: Session,
    review_id: int,
    *,
    decision: str,
    reviewer: str,
    reason: str,
    edited_output: dict[str, Any] | None = None,
) -> None:
    """Approve, edit or reject a pending item. A reason is always required; an edit needs output."""
    from datetime import UTC, datetime

    from filinglens.db.models import ReviewItem

    if decision not in REVIEW_DECISIONS:
        raise ReviewError(f"decision must be one of {sorted(REVIEW_DECISIONS)}")
    if not reason.strip() or not reviewer.strip():
        raise ReviewError("a reviewer and a reason are required")
    if decision == "edit" and edited_output is None:
        raise ReviewError("an edit needs the edited output")
    item = session.get(ReviewItem, review_id, with_for_update=True)
    if item is None:
        raise ReviewError(f"no review item {review_id}")
    if item.status != "pending":
        raise ReviewError(f"review item {review_id} is already {item.status}")
    item.status = REVIEW_DECISIONS[decision]
    item.reviewer, item.decision_reason = reviewer, reason
    item.edited_output = edited_output
    item.decided_at = datetime.now(UTC)
    audit(
        session,
        f"user:{reviewer}",
        f"review.{item.status}",
        "review_queue",
        review_id,
        {"agent_run_id": item.agent_run_id, "reason": reason, "edited": edited_output is not None},
    )


# -- Phase 8: prices -----------------------------------------------------------------------


def upsert_prices(session: Session, cik: str, bars: list[PriceBar], source: str) -> int:
    from filinglens.db.models import PriceRow

    rows: list[dict[str, object]] = [
        {
            "cik": cik,
            "day": b.day,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "source": source,
        }
        for b in bars
    ]
    for batch in _batches(rows):
        stmt = insert(PriceRow).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=[PriceRow.cik, PriceRow.day],
            set_={c: getattr(stmt.excluded, c) for c in ("open", "high", "low", "close", "volume", "source")}
            | {"fetched_at": func.now()},
        )
        session.execute(stmt)
    return len(rows)


def load_prices(session: Session, cik: str, limit: int = 100) -> list[PriceBar]:
    """The latest `limit` trading days, oldest first."""
    from sqlalchemy import select

    from filinglens.db.models import PriceRow
    from filinglens.prices.alphavantage import PriceBar

    rows = session.scalars(
        select(PriceRow).where(PriceRow.cik == cik).order_by(PriceRow.day.desc()).limit(limit)
    ).all()
    return [PriceBar(r.day, r.open, r.high, r.low, r.close, r.volume) for r in reversed(rows)]
