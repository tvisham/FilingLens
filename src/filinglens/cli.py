"""Command-line interface: `filinglens ingest` and `filinglens report ...`."""

from __future__ import annotations

import functools
import logging
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import click

from filinglens.config import load_universe
from filinglens.edgar.cache import DiskCache
from filinglens.edgar.client import SecClient, SecError
from filinglens.edgar.tickers import UnknownTickerError
from filinglens.formatting import render_cadence, render_financials
from filinglens.pipeline import load_ticker_index, persist_snapshot, snapshot_company
from filinglens.settings import get_settings

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from filinglens.agent.records import AgentRunResult
    from filinglens.edgar.source import EdgarSource
    from filinglens.risk.classify import Classifier
    from filinglens.risk.embed import Embedder
    from filinglens.risk.store import RiskStore
    from filinglens.settings import Settings

log = logging.getLogger("filinglens")


def _client() -> SecClient:
    s = get_settings()
    return SecClient(
        s.sec_user_agent,
        DiskCache(s.cache_dir),
        max_requests_per_second=s.sec_max_requests_per_second,
    )


def _as_of(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


DEFAULT_COMPANIES = Path("config/companies.yaml")


def _predecessors(ticker: str) -> list[str]:
    """Predecessor CIKs from the default universe file, so reports match `ingest`."""
    if not DEFAULT_COMPANIES.exists():
        return []
    return load_universe(DEFAULT_COMPANIES).predecessors_for(ticker)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def main(verbose: bool) -> None:
    """FilingLens: SEC filing cadence, financials and risk analysis."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@main.command()
@click.option(
    "--companies",
    "companies_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_COMPANIES,
    show_default=True,
)
@click.option("--as-of", help="Analysis date (YYYY-MM-DD). Default: today.")
@click.option("--dry-run", is_flag=True, help="Fetch and compute, but don't write to the database.")
@click.option(
    "--bulk-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Read SEC's bulk ZIPs from this folder instead of one request per company (see `bulk download`).",
)
def ingest(companies_path: Path, as_of: str | None, dry_run: bool, bulk_dir: Path | None) -> None:
    """Fetch, analyse and store every company in the universe."""
    universe = load_universe(companies_path)
    when = _as_of(as_of)
    failures: list[str] = []
    engine: Engine | None = None
    if not dry_run:
        from filinglens.db.session import make_engine, session_scope
        from filinglens.settings import get_database_url

        engine = make_engine(get_database_url())

    with _client() as client:
        index = load_ticker_index(client)
        source: EdgarSource = client
        if bulk_dir is not None:
            from filinglens.edgar.bulk import BulkSource

            source = BulkSource(bulk_dir, client)
        for spec in universe.companies():
            try:
                snap = snapshot_company(
                    source,
                    index,
                    spec.ticker,
                    sector=spec.sector,
                    as_of=when,
                    predecessor_ciks=spec.predecessor_ciks,
                )
            except (UnknownTickerError, SecError) as exc:
                failures.append(f"{spec.ticker}: {exc}")
                log.error("%s: %s", spec.ticker, exc)
                continue
            for w in snap.warnings:
                log.warning("%s: %s", spec.ticker, w)
            if engine is not None:
                from sqlalchemy.exc import SQLAlchemyError

                try:
                    with session_scope(engine) as session:  # one transaction per company
                        counts = persist_snapshot(session, snap)
                except SQLAlchemyError as exc:  # e.g. a constraint catching bad data: skip this company
                    failures.append(f"{spec.ticker}: database rejected the data: {str(exc).splitlines()[0]}")
                    log.error("%s: %s", spec.ticker, str(exc).splitlines()[0])
                    continue
                log.info("%s: stored %s", spec.ticker, counts)
            else:
                log.info("%s: ok (dry run)", spec.ticker)
        log.info("SEC requests made: %d", client.request_count)

    if failures:
        click.echo("\nFailed:\n  " + "\n  ".join(failures), err=True)
        sys.exit(1)


@main.group()
def report() -> None:
    """Print a report for one ticker (reads SEC through the cache; no database needed)."""


@report.command("cadence")
@click.argument("ticker")
@click.option("--as-of", help="Analysis date (YYYY-MM-DD). Default: today.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of text.")
def report_cadence(ticker: str, as_of: str | None, as_json: bool) -> None:
    with _client() as client:
        snap = snapshot_company(
            client,
            load_ticker_index(client),
            ticker,
            as_of=_as_of(as_of),
            predecessor_ciks=_predecessors(ticker),
        )
    click.echo(snap.cadence.model_dump_json(indent=2) if as_json else render_cadence(snap.cadence))


@report.command("financials")
@click.argument("ticker")
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of text.")
def report_financials(ticker: str, as_json: bool) -> None:
    with _client() as client:
        snap = snapshot_company(
            client,
            load_ticker_index(client),
            ticker,
            as_of=date.today(),
            predecessor_ciks=_predecessors(ticker),
        )
    if snap.financials is None:
        raise click.ClickException(f"{ticker}: no XBRL financial data on EDGAR")
    click.echo(snap.financials.model_dump_json(indent=2) if as_json else render_financials(snap.financials))


def _database_url(no_db: bool) -> str | None:
    if no_db:
        return None
    from filinglens.settings import get_database_url

    try:
        return get_database_url()
    except ValueError as exc:
        raise click.ClickException("DATABASE_URL is not set; pass --no-db to run without logging") from exc


def _log_run(database_url: str | None, run: AgentRunResult) -> int | None:
    if database_url is None:
        return None
    from filinglens.db import repository as repo
    from filinglens.db.session import make_engine, session_scope

    with session_scope(make_engine(database_url)) as session:
        run_id = repo.insert_agent_run(session, run)
        repo.audit(
            session,
            "agent",
            "agent_run.completed",
            "agent_runs",
            run_id,
            {"scenario": run.scenario, "input": run.input, "status": run.status, "cost_usd": run.cost_usd},
        )
    log.info("logged %s run %d", run.scenario, run_id)
    return run_id


def guarded_agent_run(
    scenario: str, value: str, database_url: str | None
) -> tuple[AgentRunResult, int | None]:
    """Run a scenario, apply the guardrails, log it, and queue it for review if flagged.

    Returns the run and its review-queue id (None when it passed, or without a database)."""
    import anyio

    from filinglens.agent.runner import run_scenario
    from filinglens.guardrails.review import guard, known_figures, tickers_in

    try:
        run = anyio.run(
            functools.partial(run_scenario, scenario, value, settings=get_settings(), project_root=Path.cwd())
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    with _client() as client:
        verdict = guard(run, known_figures(client, load_ticker_index(client), tickers_in(run)))
    run_id = _log_run(database_url, run)
    review_id: int | None = None
    if run_id is not None and verdict.needs_review:
        from filinglens.db import repository as repo
        from filinglens.db.session import make_engine, session_scope

        reason = "; ".join(f"[{i.severity}] {i.check}: {i.message}" for i in verdict.issues)[:4000]
        with session_scope(make_engine(database_url or "")) as session:
            review_id = repo.enqueue_review(session, run_id, reason, "block" if verdict.blocked else "review")
    return run, review_id


@main.group()
def agent() -> None:
    """Run agent scenarios (calls the Claude API and costs money)."""


def _run_agent(scenario: str, value: str, no_db: bool, as_json: bool) -> None:
    import json

    run, review_id = guarded_agent_run(scenario, value, _database_url(no_db))
    verdict = run.guardrails or {}
    if as_json:
        click.echo(json.dumps(run.to_dict() | {"review_id": review_id}, indent=2, default=str))
    else:
        click.echo(
            f"status={run.status} model={run.model} turns={run.num_turns} repairs={run.repairs} "
            f"tokens_in={run.tokens_in} tokens_out={run.tokens_out} cost_usd={run.cost_usd} "
            f"latency_ms={run.latency_ms}"
        )
        click.echo("tool calls: " + ", ".join(c["name"].rsplit("__", 1)[-1] for c in run.tool_calls))
        if run.error:
            click.echo(f"error: {run.error}", err=True)
        for issue in verdict.get("issues", []):
            click.echo(f"guardrail [{issue['severity']}] {issue['check']}: {issue['message']}", err=True)
        if verdict.get("blocked"):
            click.echo("BLOCKED: output withheld until a reviewer approves or edits it.", err=True)
        elif run.output:
            click.echo(json.dumps(run.output, indent=2))
        if review_id is not None:
            click.echo(
                f"queued for human review: #{review_id}  (filinglens review show {review_id})", err=True
            )
    if run.status != "ok" or verdict.get("blocked"):
        sys.exit(1)


_NO_DB = click.option("--no-db", is_flag=True, help="Don't log or queue the run (prints it only).")
_JSON = click.option("--json", "as_json", is_flag=True, help="Print the full run record as JSON.")


@agent.command("profile")
@click.argument("ticker")
@_NO_DB
@_JSON
def agent_profile(ticker: str, no_db: bool, as_json: bool) -> None:
    """Scenario (a): single-company profile: cadence, financials, top risk factors."""
    _run_agent("risk_profile", ticker, no_db, as_json)


@agent.command("sector")
@click.argument("sector")
@_NO_DB
@_JSON
def agent_sector(sector: str, no_db: bool, as_json: bool) -> None:
    """Scenario (b): compare risk factors across one sector of the universe."""
    _run_agent("sector_comparison", sector, no_db, as_json)


@agent.command("changes")
@click.argument("ticker")
@_NO_DB
@_JSON
def agent_changes(ticker: str, no_db: bool, as_json: bool) -> None:
    """Scenario (c): what changed in a company's risk factors year over year."""
    _run_agent("what_changed", ticker, no_db, as_json)


@agent.command("brief")
@click.argument("ticker")
@_NO_DB
@_JSON
def agent_brief(ticker: str, no_db: bool, as_json: bool) -> None:
    """Scenario (d): client-meeting prep brief."""
    _run_agent("meeting_brief", ticker, no_db, as_json)


@agent.command("ask")
@click.argument("question")
@_NO_DB
@_JSON
def agent_ask(question: str, no_db: bool, as_json: bool) -> None:
    """Ask a free-form question (answers come only from FilingLens tools)."""
    _run_agent("question", question, no_db, as_json)


# -- review queue (Phase 5) ------------------------------------------------------------


@main.group()
def review() -> None:
    """Human review of flagged agent runs: list, show, approve, edit, reject."""


def _engine() -> Engine:
    from filinglens.db.session import make_engine

    url = _database_url(False)
    assert url is not None
    return make_engine(url)


@review.command("list")
@click.option(
    "--status",
    default="pending",
    show_default=True,
    type=click.Choice(["pending", "approved", "edited", "rejected", "all"]),
)
def review_list(status: str) -> None:
    from sqlalchemy import select

    from filinglens.db.models import AgentRun, ReviewItem
    from filinglens.db.session import session_scope

    with session_scope(_engine()) as session:
        stmt = select(ReviewItem, AgentRun).join(AgentRun, AgentRun.id == ReviewItem.agent_run_id)
        if status != "all":
            stmt = stmt.where(ReviewItem.status == status)
        rows = session.execute(stmt.order_by(ReviewItem.created_at)).all()
        for item, run in rows:
            click.echo(
                f"#{item.id:<5} {item.status:9} {item.severity:6} {run.scenario:18} {run.input} "
                f"{item.created_at:%Y-%m-%d %H:%M}  {item.reason[:90]}"
            )
        click.echo(f"{len(rows)} item(s)")


@review.command("show")
@click.argument("review_id", type=int)
def review_show(review_id: int) -> None:
    import json

    from filinglens.db.models import AgentRun, ReviewItem
    from filinglens.db.session import session_scope

    with session_scope(_engine()) as session:
        item = session.get(ReviewItem, review_id)
        if item is None:
            raise click.ClickException(f"no review item {review_id}")
        run = session.get(AgentRun, item.agent_run_id)
        assert run is not None
        click.echo(f"#{item.id} {item.status} ({item.severity})  run {run.id}: {run.scenario} {run.input}")
        click.echo(f"reasons: {item.reason}")
        if item.decision_reason:
            click.echo(f"decision by {item.reviewer}: {item.decision_reason}")
        click.echo(json.dumps(item.edited_output or run.output, indent=2))


def _decide(review_id: int, decision: str, reviewer: str, reason: str, edited: Path | None) -> None:
    import json

    from filinglens.db import repository as repo
    from filinglens.db.session import session_scope

    edited_output = json.loads(edited.read_text(encoding="utf-8")) if edited else None
    try:
        with session_scope(_engine()) as session:
            repo.decide_review(
                session,
                review_id,
                decision=decision,
                reviewer=reviewer,
                reason=reason,
                edited_output=edited_output,
            )
    except repo.ReviewError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"#{review_id}: {repo.REVIEW_DECISIONS[decision]} by {reviewer}")


_REVIEWER = click.option("--reviewer", required=True, help="Your name (recorded in the audit log).")
_REASON = click.option("--reason", required=True, help="Why (recorded in the audit log).")


@review.command("approve")
@click.argument("review_id", type=int)
@_REVIEWER
@_REASON
def review_approve(review_id: int, reviewer: str, reason: str) -> None:
    _decide(review_id, "approve", reviewer, reason, None)


@review.command("reject")
@click.argument("review_id", type=int)
@_REVIEWER
@_REASON
def review_reject(review_id: int, reviewer: str, reason: str) -> None:
    _decide(review_id, "reject", reviewer, reason, None)


@review.command("edit")
@click.argument("review_id", type=int)
@click.argument("edited_json", type=click.Path(exists=True, path_type=Path))
@_REVIEWER
@_REASON
def review_edit(review_id: int, edited_json: Path, reviewer: str, reason: str) -> None:
    """Replace the output with EDITED_JSON (e.g. from `review show ID > out.json`, then edited)."""
    _decide(review_id, "edit", reviewer, reason, edited_json)


# -- risk factors (Phase 3) ------------------------------------------------------------


@main.group()
def risk() -> None:
    """Extract, embed, classify and compare 10-K risk factors."""


def _risk_store(no_db: bool) -> RiskStore:
    if no_db:
        from filinglens.risk.store import InMemoryRiskStore

        return InMemoryRiskStore()
    from filinglens.db.risk_store import SqlRiskStore
    from filinglens.db.session import make_engine

    database_url = _database_url(False)
    assert database_url is not None
    return SqlRiskStore(make_engine(database_url))


@risk.command("ingest")
@click.argument("tickers", nargs=-1)
@click.option("--all", "all_companies", is_flag=True, help="Every company in config/companies.yaml.")
@click.option(
    "--classify/--no-classify",
    default=False,
    show_default=True,
    help="Classify risk factors with Claude (costs money; logged to agent_runs).",
)
@click.option("--no-db", is_flag=True, help="Keep results in memory and print them (nothing is stored).")
@click.option("--force", is_flag=True, help="Re-extract and re-classify filings already stored.")
@click.option("--hashing-embedder", is_flag=True, hidden=True, help="Test only: skip the bge model.")
@click.option(
    "--companies",
    "companies_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Universe file for --all / --sample (default config/companies.yaml).",
)
@click.option("--sample", type=int, help="Classify a random sample of N companies from the universe.")
@click.option("--seed", type=int, default=0, show_default=True, help="Seed for --sample (reproducible).")
@click.option(
    "--max-cost", type=float, help="Stop classifying once the estimated spend would pass this (USD)."
)
def risk_ingest(
    tickers: tuple[str, ...],
    all_companies: bool,
    classify: bool,
    no_db: bool,
    force: bool,
    hashing_embedder: bool,
    companies_path: Path | None,
    sample: int | None,
    seed: int,
    max_cost: float | None,
) -> None:
    """Ingest the latest two 10-Ks' risk factors for TICKERS (or --all / --sample N)."""
    from filinglens.risk.ingest import ingest_company_risks

    universe_path = companies_path or DEFAULT_COMPANIES
    universe = load_universe(universe_path) if universe_path.exists() else None
    if all_companies or sample:
        if universe is None:
            raise click.ClickException(f"{universe_path} not found")
        wanted = [c.ticker for c in universe.companies()]
        if sample:
            import random

            wanted = sorted(random.Random(seed).sample(wanted, min(sample, len(wanted))))
            click.echo(f"sample of {len(wanted)} (seed {seed}): {' '.join(wanted)}")
    else:
        wanted = [t.upper() for t in tickers]
    if not wanted:
        raise click.ClickException("give one or more tickers, --all or --sample N")

    settings = get_settings()
    store = _risk_store(no_db)
    database_url = None if no_db else _database_url(False)
    embedder = _embedder(hashing_embedder)
    classifier = _classifier(settings) if classify else None
    failures: list[str] = []
    total_cost = Decimal(0)
    with _client() as client:
        index = load_ticker_index(client)
        for ticker in wanted:
            spec = next((c for c in universe.companies() if c.ticker == ticker), None) if universe else None
            try:
                res = ingest_company_risks(
                    client,
                    store,
                    embedder,
                    index,
                    ticker,
                    sector=spec.sector if spec else "unknown",
                    predecessor_ciks=spec.predecessor_ciks if spec else (),
                    classifier=classifier,
                    log_run=functools.partial(_log_run, database_url),
                    force=force,
                    max_cost_usd=Decimal(str(max_cost)) if max_cost is not None else None,
                )
            except (UnknownTickerError, SecError) as exc:
                failures.append(f"{ticker}: {exc}")
                log.error("%s: %s", ticker, exc)
                continue
            for s in res.sections:
                click.echo(
                    f"{res.ticker:5} FY{s.fiscal_year} {s.accession} {s.method}/{s.split_method} "
                    f"conf={s.confidence} factors={s.factors} chunks={s.chunks}"
                    + (" (reused)" if s.reused else "")
                    + (" NEEDS REVIEW" if s.needs_review else "")
                )
            if res.classification is not None:
                c = res.classification
                total_cost += c.cost_usd
                click.echo(
                    f"      classified={len(c.results)} failed={len(c.failures)} requests={c.requests} "
                    f"tokens={c.tokens_in}/{c.tokens_out} cost=${c.cost_usd:.4f} "
                    f"served_by={sorted(c.served_by)}"
                )
            if res.diff_counts:
                click.echo(f"      diff: {res.diff_counts}")
            for w in res.warnings:
                log.warning("%s: %s", res.ticker, w)
            if no_db:
                _print_factors(store, res.cik)
    if classifier is not None:
        click.echo(f"total classification cost: ${total_cost:.4f}")
    if failures:
        click.echo("\nFailed:\n  " + "\n  ".join(failures), err=True)
        sys.exit(1)


def _embedder(hashing: bool) -> Embedder:
    if hashing:
        from filinglens.risk.embed import HashingEmbedder

        return HashingEmbedder()
    from filinglens.risk.embed import BgeSmallEmbedder

    try:
        return BgeSmallEmbedder()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


def _classifier(settings: Settings) -> Classifier:
    import anthropic

    from filinglens.risk.classify import Classifier

    client = (
        anthropic.Anthropic(api_key=settings.anthropic_api_key)
        if settings.anthropic_api_key
        else anthropic.Anthropic()
    )
    return Classifier(
        client,
        model=settings.classifier_model,
        effort=settings.classifier_effort,
        max_factors_per_request=settings.classifier_factors_per_request,
    )


def _print_factors(store: RiskStore, cik: str, fiscal_year: int | None = None) -> None:
    for section in store.sections(cik):
        if fiscal_year is not None and section.fiscal_year != fiscal_year:
            continue
        click.echo(f"\n  FY{section.fiscal_year} {section.source_url}")
        for f in store.factors(section.id):
            tags = (
                f"[{', '.join(f.categories)}] sev={f.severity}"
                if f.status == "classified"
                else f"({f.status})"
            )
            click.echo(f"   {f.ordinal:3}. {(f.heading or '<window>')[:110]}")
            click.echo(
                f"        {tags} chunks={f.chunk_ids}"
                + (f" cited={f.cited_chunk_ids}" if f.cited_chunk_ids else "")
            )
            if f.summary:
                click.echo(f"        {f.summary}")
        if fiscal_year is None:
            break  # latest year only by default


@risk.command("classify")
@click.argument("tickers", nargs=-1)
@click.option("--all", "all_companies", is_flag=True, help="Every company in config/companies.yaml.")
@click.option(
    "--local",
    "local",
    is_flag=True,
    required=True,
    help="Free local classifier (embedding zero-shot categories, extractive summaries; no LLM).",
)
@click.option("--force", is_flag=True, help="Re-classify factors that are already classified.")
@click.option("--hashing-embedder", is_flag=True, hidden=True, help="Test only: skip the bge model.")
def risk_classify(
    tickers: tuple[str, ...], all_companies: bool, local: bool, force: bool, hashing_embedder: bool
) -> None:
    """Classify stored risk factors (both stored 10-Ks per company) without re-extracting them."""
    from filinglens.agent.records import AgentRunResult
    from filinglens.risk.classify import FactorInput
    from filinglens.risk.local_classify import LOCAL_MODEL, LocalClassifier

    universe = load_universe(DEFAULT_COMPANIES) if DEFAULT_COMPANIES.exists() else None
    wanted = (
        [c.ticker for c in universe.companies()]
        if all_companies and universe
        else [t.upper() for t in tickers]
    )
    if not wanted:
        raise click.ClickException("give one or more tickers, or --all")
    database_url = _database_url(False)
    store = _risk_store(False)
    classifier = LocalClassifier(_embedder(hashing_embedder))
    with _client() as client:
        index = load_ticker_index(client)
        for ticker in wanted:
            entry = index.resolve(ticker)
            factors = [
                f
                for s in store.sections(entry.cik)
                for f in store.factors(s.id)
                if force or f.status != "classified"
            ]
            if not factors:
                click.echo(f"{entry.ticker:5} nothing to classify (run `risk ingest {entry.ticker}` first?)")
                continue
            run = classifier.classify(
                [FactorInput(f"F{f.id}", f.heading, f.text, tuple(f.chunk_ids)) for f in factors]
            )
            record = AgentRunResult(
                scenario="classify_risk_factors_local",
                input={"ticker": entry.ticker, "factor_ids": [f.id for f in factors]},
                status=run.status,
                output={"classified": len(run.results), "failed": len(run.failures)},
                model=LOCAL_MODEL,
                tokens_in=0,
                tokens_out=0,
                cost_usd=Decimal(0),
                latency_ms=run.latency_ms,
            )
            run_id = _log_run(database_url, record)
            store.save_classifications(
                {int(k[1:]): c for k, c in run.results.items()},
                {int(k[1:]): reason for k, reason in run.failures.items()},
                model=LOCAL_MODEL,
                agent_run_id=run_id,
            )
            counts: dict[str, int] = {}
            for c in run.results.values():
                counts[str(c.categories[0])] = counts.get(str(c.categories[0]), 0) + 1
            top = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:4])
            click.echo(
                f"{entry.ticker:5} classified {len(run.results)} factors "
                f"({run.latency_ms} ms; $0)  top: {top}"
            )


@risk.command("report")
@click.argument("tickers", nargs=-1)
@click.option("--all", "all_companies", is_flag=True, help="Every company in config/companies.yaml.")
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("reports/risk-summaries.md"), show_default=True
)
def risk_report(tickers: tuple[str, ...], all_companies: bool, out: Path) -> None:
    """Write a Markdown report: each company's latest risk factors, category mix and changes."""
    from collections import Counter

    from filinglens.risk.classify import Category
    from filinglens.risk.local_classify import LOCAL_MODEL

    universe = load_universe(DEFAULT_COMPANIES) if DEFAULT_COMPANIES.exists() else None
    specs = {c.ticker: c.sector for c in universe.companies()} if universe else {}
    wanted = list(specs) if all_companies else [t.upper() for t in tickers]
    if not wanted:
        raise click.ClickException("give one or more tickers, or --all")
    store = _risk_store(False)
    label = {c.value: c.value.replace("_", " ") for c in Category}
    lines: list[str] = []
    models: set[str] = set()
    toc: list[str] = []
    with _client() as client:
        index = load_ticker_index(client)
        for ticker in wanted:
            entry = index.resolve(ticker)
            sections = store.sections(entry.cik)
            if not sections:
                lines += [f"## {entry.ticker}", "", "_No risk factors ingested._", ""]
                continue
            latest = sections[0]
            factors = store.factors(latest.id)
            models |= {f.model for f in factors if f.model}
            mix = Counter(f.categories[0] for f in factors if f.categories)
            toc.append(f"[{entry.ticker}](#{entry.ticker.lower()})")
            lines += [
                f"## {entry.ticker}",
                "",
                f"**{entry.title}** · {specs.get(entry.ticker, '')} · FY{latest.fiscal_year} 10-K "
                f"([filing]({latest.source_url})) · {len(factors)} risk factors"
                + (" · **extraction flagged for review**" if latest.needs_review else ""),
                "",
                "**Main risk areas:** "
                + ", ".join(f"{label.get(c, c)} ({n})" for c, n in mix.most_common(4)),
                "",
            ]
            diffs = store.diffs(entry.cik)
            if diffs:
                by_id = {f.id: f for s in sections[:2] for f in store.factors(s.id)}
                new = [by_id[d.to_factor_id] for d in diffs if d.status == "new" and d.to_factor_id in by_id]
                gone = [
                    by_id[d.from_factor_id]
                    for d in diffs
                    if d.status == "removed" and d.from_factor_id in by_id
                ]
                counts = Counter(d.status for d in diffs)
                lines.append(
                    f"**Since FY{diffs[0].from_year}:** {counts['new']} new, {counts['removed']} removed, "
                    f"{counts['changed']} reworded, {counts['unchanged']} unchanged."
                )
                lines += [f"- New: {(f.summary or f.heading or '')[:200]}" for f in new[:5]]
                lines += [f"- Removed: {(f.summary or f.heading or '')[:200]}" for f in gone[:5]]
                lines.append("")
            lines += ["| # | Risk (summary) | Categories |", "|---|---|---|"]
            for f in factors:
                text = (f.summary or f.heading or "(untitled)").replace("|", "/").replace("\n", " ")
                cats = ", ".join(label.get(c, c) for c in f.categories) or f.status
                lines.append(f"| {f.ordinal} | {text} | {cats} |")
            lines.append("")
    local = LOCAL_MODEL in models
    header = [
        "# Risk-factor summaries",
        "",
        f"From each company's latest 10-K Item 1A, generated {date.today()}. " + ", ".join(toc),
        "",
        (
            "> **How these were made:** categories by local embedding similarity and summaries taken "
            "from each risk factor's own heading or first sentence (no LLM). They are a free "
            "baseline, not an analyst's reading; severity is not assessed."
            if local
            else "> Summaries and categories by the configured classifier."
        ),
        "> For information only. Not investment advice.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + lines), encoding="utf-8")
    click.echo(f"wrote {out} ({len(toc)} companies)")


@risk.command("show")
@click.argument("ticker")
@click.option("--year", "fiscal_year", type=int, help="Fiscal year (default: latest).")
def risk_show(ticker: str, fiscal_year: int | None) -> None:
    """Print stored risk factors for TICKER."""
    store = _risk_store(False)
    with _client() as client:
        entry = load_ticker_index(client).resolve(ticker)
    _print_factors(store, entry.cik, fiscal_year)


@risk.command("changes")
@click.argument("ticker")
def risk_changes(ticker: str) -> None:
    """Print the stored year-over-year risk-factor diff for TICKER."""
    store = _risk_store(False)
    with _client() as client:
        entry = load_ticker_index(client).resolve(ticker)
    rows = store.diffs(entry.cik)
    if not rows:
        raise click.ClickException(f"no diff stored for {ticker}; ingest two 10-Ks first")
    by_id = {f.id: f for s in store.sections(entry.cik) for f in store.factors(s.id)}
    click.echo(f"{entry.ticker}: FY{rows[0].from_year} -> FY{rows[0].to_year}")
    for r in rows:
        if r.status == "unchanged":
            continue
        f = by_id.get(r.to_factor_id or r.from_factor_id or -1)
        sim = f" text_sim={r.text_similarity}" if r.text_similarity is not None else ""
        click.echo(f"  {r.status:8}{sim} {(f.heading if f else '?') or '<window>'}"[:160])
    click.echo(f"  unchanged: {sum(r.status == 'unchanged' for r in rows)}")


@main.group()
def api() -> None:
    """The web API (FastAPI)."""


@api.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def api_serve(host: str, port: int) -> None:
    """Serve the API for the web app (http://127.0.0.1:8000/docs)."""
    import uvicorn

    from filinglens.api.app import default_app

    uvicorn.run(default_app(), host=host, port=port)


@api.command("openapi")
@click.option("--out", type=click.Path(path_type=Path), default=Path("web/openapi.json"), show_default=True)
def api_openapi(out: Path) -> None:
    """Write the OpenAPI schema (the web app generates its typed client from it)."""
    import json

    from filinglens.api.app import Backend, create_app

    spec = create_app(Backend(lambda: None, None)).openapi()  # type: ignore[arg-type, return-value]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        print(json.dumps(spec, indent=2), file=fh)
    click.echo(f"wrote {out} ({len(spec['paths'])} paths)")


@main.group()
def bulk() -> None:
    """SEC's nightly bulk archives (companyfacts.zip, submissions.zip) for large universes."""


@bulk.command("download")
@click.option(
    "--dir",
    "directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/bulk"),
    show_default=True,
)
@click.option("--yes", is_flag=True, help="Don't ask before downloading (several GB).")
def bulk_download(directory: Path, yes: bool) -> None:
    """Download companyfacts.zip and submissions.zip (documented SEC bulk data, several GB)."""
    from filinglens.edgar.bulk import BULK_URLS, download_bulk

    if not yes:
        click.confirm(
            "This downloads SEC's two nightly bulk archives (each over a gigabyte):\n  "
            + "\n  ".join(BULK_URLS.values())
            + f"\ninto {directory}. Continue?",
            abort=True,
        )
    with _client() as client:
        sizes = download_bulk(client, directory)
    for name, size in sizes.items():
        click.echo(f"{name}.zip: {size / 1e9:.2f} GB")


@main.group()
def universe() -> None:
    """Build company universes."""


@universe.command("build")
@click.option("--top", type=int, default=500, show_default=True)
@click.option(
    "--bulk-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/bulk"),
    show_default=True,
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("config/universe_500.yaml"), show_default=True
)
def universe_build(top: int, bulk_dir: Path, out: Path) -> None:
    """Rank companies by their latest reported public float (dei:EntityPublicFloat) and write
    the top N, grouped by SIC division, as a universe file for `ingest --companies`."""
    import yaml

    from filinglens.edgar.bulk import BulkSource, public_float_ranking, sic_sector

    with _client() as client, BulkSource(bulk_dir, client) as source:
        ranked = public_float_ranking(source, top)
    sectors: dict[str, list[str]] = {}
    seen: set[str] = set()
    for r in ranked:
        if r.ticker in seen:
            continue
        seen.add(r.ticker)
        sectors.setdefault(sic_sector(r.sic), []).append(r.ticker)
    header = (
        f"# Generated by `filinglens universe build --top {top}` on {date.today()}.\n"
        "# Ranked by the latest dei:EntityPublicFloat each company reported (market value of\n"
        "# non-affiliate shares, from its 10-K cover page); sectors are SEC SIC divisions.\n"
        "# Review before use: add `predecessors:` for any successor registrants.\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump({"sectors": sectors}, sort_keys=True, width=100), encoding="utf-8")
    click.echo(f"wrote {out}: {len(seen)} companies in {len(sectors)} sectors")
    if ranked:
        click.echo(
            f"largest: {ranked[0].ticker} ${ranked[0].public_float_usd / 1e9:,.0f}B; "
            f"smallest included: {ranked[-1].ticker} ${ranked[-1].public_float_usd / 1e9:,.1f}B"
        )


@main.group()
def prices() -> None:
    """End-of-day stock prices (Alpha Vantage, free key). Factual context, never advice."""


@prices.command("fetch")
@click.argument("tickers", nargs=-1)
@click.option("--all", "all_companies", is_flag=True, help="Every company in config/companies.yaml.")
def prices_fetch(tickers: tuple[str, ...], all_companies: bool) -> None:
    """Fetch the latest ~100 trading days and store them (one request per company; 25/day free)."""
    from filinglens.db import repository as repo
    from filinglens.db.session import make_engine, session_scope
    from filinglens.edgar.cache import DiskCache
    from filinglens.prices.alphavantage import SOURCE, AlphaVantageClient, PriceSourceError

    settings = get_settings()
    if not settings.alphavantage_api_key:
        raise click.ClickException(
            "ALPHAVANTAGE_API_KEY is not set in .env (free key: https://www.alphavantage.co/support/#api-key)"
        )
    universe = load_universe(DEFAULT_COMPANIES) if DEFAULT_COMPANIES.exists() else None
    wanted = (
        [c.ticker for c in universe.companies()]
        if all_companies and universe
        else [t.upper() for t in tickers]
    )
    if not wanted:
        raise click.ClickException("give one or more tickers, or --all")
    database_url = _database_url(False)
    assert database_url is not None
    engine = make_engine(database_url)
    stored = 0
    with (
        _client() as sec,
        AlphaVantageClient(settings.alphavantage_api_key, DiskCache(settings.cache_dir)) as av,
    ):
        index = load_ticker_index(sec)
        for ticker in wanted:
            entry = index.resolve(ticker)
            try:
                bars = av.daily(entry.ticker)
            except PriceSourceError as exc:
                click.echo(f"{entry.ticker:5} stopped: {exc}", err=True)
                if "limit" in str(exc).lower() or "api key" in str(exc).lower():
                    click.echo(
                        "Quota or key problem: remaining tickers skipped; run again tomorrow.", err=True
                    )
                    break
                continue
            with session_scope(engine) as session:
                if session.get(_company_model(), entry.cik) is None:
                    click.echo(
                        f"{entry.ticker:5} skipped: run `filinglens ingest` first (company not stored)",
                        err=True,
                    )
                    continue
                n = repo.upsert_prices(session, entry.cik, bars, SOURCE)
            stored += 1
            click.echo(f"{entry.ticker:5} {n} days to {bars[-1].day}, last close ${bars[-1].close}")
        click.echo(f"{stored} companies updated; Alpha Vantage requests used: {av.request_count}")


def _company_model() -> type:
    from filinglens.db.models import Company

    return Company


@prices.command("show")
@click.argument("ticker")
@click.option("--days", default=10, show_default=True)
def prices_show(ticker: str, days: int) -> None:
    """Show stored closes and the approximate market capitalisation."""
    from filinglens.mcp_server.server import default_context
    from filinglens.mcp_server.tools import get_price_history

    result = get_price_history(default_context(), ticker, days=days)
    if result.status != "available":
        raise click.ClickException(result.message)
    for p in result.closes:
        click.echo(f"  {p.day}  ${p.close:>10}  vol {p.volume:,}")
    if result.market_cap_usd is not None:
        click.echo(
            f"approx. market cap ${result.market_cap_usd / 10**9:,.1f}B "
            f"({result.shares_outstanding:,} shares as of {result.shares_as_of}, {result.shares_source})"
        )
    click.echo(result.source)


def _register_eval() -> None:
    from filinglens.evals.cli import eval_group

    main.add_command(eval_group)


_register_eval()


if __name__ == "__main__":
    main()
