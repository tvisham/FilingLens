"""FastAPI backend for the web app (Phase 6).

Company data comes from the same tool functions the MCP server exposes, so the web app and
the agent see identical, deterministic numbers. Evals come from the database, or from the
JSON reports in `evals/reports/` when there is none. The review queue needs the database.

Everything is read-only except `POST /api/review/{id}/decision`, which requires the
`X-API-Token` header when `API_TOKEN` is set (always set it outside local development).

Run: uv run uvicorn filinglens.api.app:app --reload
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Engine, select

from filinglens.mcp_server import tools
from filinglens.mcp_server.schemas import (
    ChunkOut,
    CompanyInfo,
    FilingIndex,
    FinancialsResult,
    PriceHistoryResult,
    RiskChangesResult,
    RiskFactorsResult,
)

REPORTS = Path("evals/reports")
TICKER = r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$"


# -- response models -------------------------------------------------------------------


class CompanyListItem(BaseModel):
    ticker: str
    sector: str


class CompanyDetail(BaseModel):
    company: CompanyInfo
    filings: FilingIndex
    financials: FinancialsResult
    risks: RiskFactorsResult
    changes: RiskChangesResult
    prices: PriceHistoryResult


class ChunkDetail(ChunkOut):
    filing_url: str


class MetricPoint(BaseModel):
    metric: str
    value: float | None
    passed: bool | None
    threshold: dict[str, float]
    n_cases: int


class EvalRunSummary(BaseModel):
    id: str
    suite: str
    started_at: str
    passed: bool
    cost_usd: float
    git_sha: str | None
    metrics: list[MetricPoint]


class EvalCase(BaseModel):
    case_id: str
    metric: str
    score: float | None
    passed: bool | None
    details: dict[str, Any]


class EvalRunDetail(EvalRunSummary):
    failed_cases: list[EvalCase]


class ReviewItemOut(BaseModel):
    id: int
    agent_run_id: int
    scenario: str
    input: dict[str, Any]
    status: str
    severity: str
    reason: str
    reviewer: str | None
    decision_reason: str | None
    created_at: str
    decided_at: str | None


class ReviewItemDetail(ReviewItemOut):
    output: dict[str, Any] | None
    edited_output: dict[str, Any] | None
    guardrails: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    cost_usd: float | None


class Decision(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    reviewer: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=2000)
    edited_output: dict[str, Any] | None = None


# -- dependencies ------------------------------------------------------------------------


class Backend:
    def __init__(
        self,
        ctx_factory: Callable[[], tools.ToolContext],
        engine: Engine | None,
        *,
        api_token: str | None = None,
        reports_dir: Path = REPORTS,
    ) -> None:
        self._ctx_factory = ctx_factory
        self._ctx: tools.ToolContext | None = None
        self.engine = engine
        self.api_token = api_token
        self.reports_dir = reports_dir

    @property
    def ctx(self) -> tools.ToolContext:
        if self._ctx is None:
            self._ctx = self._ctx_factory()
        return self._ctx


def get_backend(request: Request) -> Backend:
    backend: Backend = request.app.state.backend
    return backend


BackendDep = Annotated[Backend, Depends(get_backend)]
TickerPath = Annotated[str, PathParam(pattern=TICKER)]


def _http(exc: tools.ToolInputError) -> HTTPException:
    return HTTPException(status_code=404 if "unknown" in str(exc) else 400, detail=str(exc))


def create_app(backend: Backend, cors_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="FilingLens API", version="0.1.0")
    app.state.backend = backend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Token"],
    )

    def engine_of(b: Backend) -> Engine:
        if b.engine is None:
            raise HTTPException(status_code=503, detail="the database is not configured (DATABASE_URL)")
        return b.engine

    @app.get("/api/health")
    def health(b: BackendDep) -> dict[str, Any]:
        return {"ok": True, "database": b.engine is not None, "risk_store": b.ctx.store is not None}

    @app.get("/api/companies")
    def companies(b: BackendDep) -> list[CompanyListItem]:
        universe = b.ctx.universe
        return (
            [CompanyListItem(ticker=c.ticker, sector=c.sector) for c in universe.companies()]
            if universe
            else []
        )

    @app.get("/api/companies/{ticker}")
    def company(ticker: TickerPath, b: BackendDep) -> CompanyDetail:
        ctx = b.ctx
        try:
            info = tools.lookup_company(ctx, ticker)
            filings = tools.get_filing_index(ctx, ticker, forms=["10-K", "10-Q", "8-K", "DEF 14A"], limit=60)
            financials = tools.get_financials(ctx, ticker, quarters=8)
        except tools.ToolInputError as exc:
            raise _http(exc) from exc
        try:
            risks = tools.get_risk_factors(ctx, ticker)
            changes = tools.get_risk_changes(ctx, ticker)
        except tools.ToolInputError as exc:  # no risk store configured
            risks = RiskFactorsResult(ticker=info.ticker, status="not_available", message=str(exc))
            changes = RiskChangesResult(ticker=info.ticker, status="not_available", message=str(exc))
        prices = tools.get_price_history(ctx, ticker, days=100)
        return CompanyDetail(
            company=info, filings=filings, financials=financials, risks=risks, changes=changes, prices=prices
        )

    @app.get("/api/chunks/{chunk_id}")
    def chunk(chunk_id: int, b: BackendDep) -> ChunkDetail:
        try:
            result = tools.get_chunks(b.ctx, [chunk_id])
        except tools.ToolInputError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not result.chunks:
            raise HTTPException(status_code=404, detail=f"no chunk {chunk_id}")
        c = result.chunks[0]
        return ChunkDetail(**c.model_dump(), filing_url=c.source_url)

    # -- evals ----------------------------------------------------------------------------

    def _summary_from_report(rid: str, r: dict[str, Any]) -> EvalRunSummary:
        return EvalRunSummary(
            id=rid,
            suite=r["suite"],
            started_at=r["started_at"],
            passed=bool(r.get("passed")),
            cost_usd=float(r.get("cost_usd") or 0),
            git_sha=r.get("git_sha"),
            metrics=[
                MetricPoint(
                    metric=m["metric"],
                    value=m["value"],
                    passed=m["passed"],
                    threshold=m["threshold"],
                    n_cases=m["n_cases"],
                )
                for m in r["metrics"]
            ],
        )

    def _reports(b: Backend) -> list[tuple[str, dict[str, Any]]]:
        if not b.reports_dir.exists():
            return []
        out = []
        for path in sorted(b.reports_dir.glob("*.json")):
            try:
                out.append((f"file:{path.stem}", json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        return out

    @app.get("/api/evals")
    def evals(b: BackendDep, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[EvalRunSummary]:
        if b.engine is None:
            runs = [_summary_from_report(rid, r) for rid, r in _reports(b)]
            return sorted(runs, key=lambda r: r.started_at)[-limit:]
        from filinglens.db.models import EvalResult, EvalRun
        from filinglens.db.session import session_scope

        with session_scope(b.engine) as session:
            rows = session.scalars(select(EvalRun).order_by(EvalRun.started_at.desc()).limit(limit)).all()
            metrics = session.scalars(
                select(EvalResult).where(
                    EvalResult.kind == "metric", EvalResult.eval_run_id.in_([r.id for r in rows])
                )
            ).all()
            by_run: dict[int, list[MetricPoint]] = {}
            for m in metrics:
                by_run.setdefault(m.eval_run_id, []).append(
                    MetricPoint(
                        metric=m.metric,
                        value=m.score,
                        passed=m.passed,
                        threshold=m.threshold or {},
                        n_cases=int(m.details.get("n_cases", 0)),
                    )
                )
            return [
                EvalRunSummary(
                    id=f"db:{r.id}",
                    suite=r.suite,
                    started_at=r.started_at.isoformat(),
                    passed=r.passed,
                    cost_usd=float(r.cost_usd),
                    git_sha=r.git_sha,
                    metrics=by_run.get(r.id, []),
                )
                for r in reversed(rows)
            ]

    @app.get("/api/evals/{run_id}")
    def eval_run(run_id: str, b: BackendDep) -> EvalRunDetail:
        if run_id.startswith("file:"):
            match = next((r for rid, r in _reports(b) if rid == run_id), None)
            if match is None:
                raise HTTPException(status_code=404, detail=f"no eval report {run_id}")
            summary = _summary_from_report(run_id, match)
            failed = [EvalCase(**c) for c in match["cases"] if c["passed"] is False]
            return EvalRunDetail(**summary.model_dump(), failed_cases=failed)
        if not run_id.startswith("db:") or not run_id[3:].isdigit():
            raise HTTPException(status_code=404, detail=f"no eval run {run_id}")
        from filinglens.db.models import EvalResult, EvalRun
        from filinglens.db.session import session_scope

        with session_scope(engine_of(b)) as session:
            run = session.get(EvalRun, int(run_id[3:]))
            if run is None:
                raise HTTPException(status_code=404, detail=f"no eval run {run_id}")
            results = session.scalars(select(EvalResult).where(EvalResult.eval_run_id == run.id)).all()
            return EvalRunDetail(
                id=run_id,
                suite=run.suite,
                started_at=run.started_at.isoformat(),
                passed=run.passed,
                cost_usd=float(run.cost_usd),
                git_sha=run.git_sha,
                metrics=[
                    MetricPoint(
                        metric=m.metric,
                        value=m.score,
                        passed=m.passed,
                        threshold=m.threshold or {},
                        n_cases=int(m.details.get("n_cases", 0)),
                    )
                    for m in results
                    if m.kind == "metric"
                ],
                failed_cases=[
                    EvalCase(
                        case_id=c.case_id or "",
                        metric=c.metric,
                        score=c.score,
                        passed=c.passed,
                        details=c.details,
                    )
                    for c in results
                    if c.kind == "case" and c.passed is False
                ],
            )

    # -- review queue ---------------------------------------------------------------------

    def _item(item: Any, run: Any) -> dict[str, Any]:
        return {
            "id": item.id,
            "agent_run_id": item.agent_run_id,
            "scenario": run.scenario,
            "input": run.input,
            "status": item.status,
            "severity": item.severity,
            "reason": item.reason,
            "reviewer": item.reviewer,
            "decision_reason": item.decision_reason,
            "created_at": item.created_at.isoformat(),
            "decided_at": item.decided_at.isoformat() if item.decided_at else None,
        }

    @app.get("/api/review")
    def review_list(
        b: BackendDep,
        status: Annotated[Literal["pending", "approved", "edited", "rejected", "all"], Query()] = "pending",
    ) -> list[ReviewItemOut]:
        from filinglens.db.models import AgentRun, ReviewItem
        from filinglens.db.session import session_scope

        with session_scope(engine_of(b)) as session:
            stmt = select(ReviewItem, AgentRun).join(AgentRun, AgentRun.id == ReviewItem.agent_run_id)
            if status != "all":
                stmt = stmt.where(ReviewItem.status == status)
            rows = session.execute(stmt.order_by(ReviewItem.created_at)).all()
            return [ReviewItemOut(**_item(item, run)) for item, run in rows]

    @app.get("/api/review/{review_id}")
    def review_detail(review_id: int, b: BackendDep) -> ReviewItemDetail:
        from filinglens.db.models import AgentRun, ReviewItem
        from filinglens.db.session import session_scope

        with session_scope(engine_of(b)) as session:
            item = session.get(ReviewItem, review_id)
            if item is None:
                raise HTTPException(status_code=404, detail=f"no review item {review_id}")
            run = session.get(AgentRun, item.agent_run_id)
            if run is None:  # FK cascade makes this impossible, but don't 500 if it happens
                raise HTTPException(status_code=404, detail=f"agent run {item.agent_run_id} not found")
            return ReviewItemDetail(
                **_item(item, run),
                output=run.output,
                edited_output=item.edited_output,
                guardrails=run.guardrails,
                tool_calls=run.tool_calls,
                cost_usd=float(run.cost_usd) if run.cost_usd is not None else None,
            )

    def require_token(b: BackendDep, x_api_token: Annotated[str | None, Header()] = None) -> None:
        if b.api_token and not (x_api_token and secrets.compare_digest(x_api_token, b.api_token)):
            raise HTTPException(status_code=401, detail="a valid X-API-Token header is required")

    @app.post("/api/review/{review_id}/decision", dependencies=[Depends(require_token)])
    def review_decide(review_id: int, decision: Decision, b: BackendDep) -> ReviewItemDetail:
        from filinglens.db import repository as repo
        from filinglens.db.session import session_scope

        try:
            with session_scope(engine_of(b)) as session:
                repo.decide_review(
                    session,
                    review_id,
                    decision=decision.decision,
                    reviewer=decision.reviewer,
                    reason=decision.reason,
                    edited_output=decision.edited_output,
                )
        except repo.ReviewError as exc:
            code = 404 if "no review item" in str(exc) else 409 if "already" in str(exc) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        return review_detail(review_id, b)

    return app


def default_app() -> FastAPI:
    from filinglens.mcp_server.server import default_context
    from filinglens.settings import get_settings

    settings = get_settings()
    engine = None
    if settings.database_url:
        from filinglens.db.session import make_engine

        engine = make_engine(settings.database_url)
    return create_app(
        Backend(default_context, engine, api_token=settings.api_token), settings.api_cors_origins
    )


def __getattr__(name: str) -> Any:  # `uvicorn filinglens.api.app:app` builds the app lazily
    if name == "app":
        return default_app()
    raise AttributeError(name)
