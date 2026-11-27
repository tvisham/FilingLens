"""`filinglens eval ...`: run suites, snapshot golden candidates, hand-label."""

from __future__ import annotations

import functools
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import click

from filinglens.config import load_universe
from filinglens.evals import golden
from filinglens.evals.results import EvalReport, load_thresholds

COMPANIES = Path("config/companies.yaml")
REPORTS = Path("evals/reports")
AGENT_PLAN = (
    ("risk_profile", "AAPL"),
    ("risk_profile", "JPM"),
    ("risk_profile", "XOM"),
    ("what_changed", "AAPL"),
    ("sector_comparison", "energy"),
)


@click.group("eval")
def eval_group() -> None:
    """Evaluations: `run` a suite, `snapshot` golden candidates, `label` by hand."""


def _render(report: EvalReport) -> str:
    lines = [
        f"{report.suite} suite  {report.started_at}  git={report.git_sha or '-'}  "
        f"cost=${report.cost_usd:.4f}",
        "",
    ]
    for m in report.metrics:
        flag = {True: "PASS", False: "FAIL", None: "skip"}[m.passed]
        value = "-" if m.value is None else f"{m.value:g}"
        bound = " ".join(f"{k}={v:g}" for k, v in m.threshold.items()) or "no threshold"
        extra = f"  failed: {', '.join(m.details['failed'][:6])}" if m.details.get("failed") else ""
        lines.append(f"  {flag}  {m.metric:24} {value:>10}  ({bound}; n={m.n_cases}){extra}")
    lines += ["", "GATE: " + ("PASSED" if report.passed else "FAILED")]
    return "\n".join(lines)


@eval_group.command("run")
@click.option("--suite", type=click.Choice(["smoke", "full"]), default="smoke", show_default=True)
@click.option("--no-db", is_flag=True, help="Don't store results in eval_runs / eval_results.")
@click.option("--report-dir", type=click.Path(path_type=Path), default=REPORTS, show_default=True)
@click.option("--yes", is_flag=True, help="Don't ask before the full suite spends money.")
@click.option(
    "--all-companies", is_flag=True, help="Smoke suite over every company, not the 5-company PR sample."
)
def eval_run(suite: str, no_db: bool, report_dir: Path, yes: bool, all_companies: bool) -> None:
    """Run an eval suite; exit 1 if the gate fails."""
    import anyio

    from filinglens.cli import _client, _database_url, _log_run
    from filinglens.evals.runner import EvalContext, run_suite
    from filinglens.mcp_server.server import build_server, default_context
    from filinglens.pipeline import load_ticker_index
    from filinglens.settings import get_settings

    settings = get_settings()
    database_url = _database_url(no_db)
    universe = load_universe(COMPANIES)
    if suite == "full" and not yes:
        click.confirm(
            "The full suite calls Claude (classifier, judge and agent runs; roughly a few dollars, "
            f"agent runs capped at ${settings.agent_max_budget_usd:g} each). Continue?",
            abort=True,
        )
    with _client() as client:
        ctx = EvalContext(
            client=client,
            index=load_ticker_index(client),
            universe=universe,
            thresholds=load_thresholds(),
            server_factory=lambda: build_server(default_context),
            log=lambda m: click.echo(f"  .. {m}", err=True),
        )
        if all_companies:
            ctx.smoke_tickers = tuple(c.ticker for c in universe.companies())
        if database_url is not None:
            from filinglens.db.risk_store import SqlRiskStore
            from filinglens.db.session import make_engine

            ctx.risk_store = SqlRiskStore(make_engine(database_url))
        if golden.load(golden.RiskLabel, golden.RISK_LABELS):
            try:  # needs the local embedding model (`uv sync --extra embeddings`)
                from filinglens.risk.embed import BgeSmallEmbedder
                from filinglens.risk.local_classify import LocalClassifier

                ctx.local_classifier = LocalClassifier(BgeSmallEmbedder())
            except RuntimeError as exc:
                click.echo(f"  .. local classifier skipped: {exc}", err=True)
        if suite == "full":
            import anthropic

            from filinglens.agent.runner import run_scenario
            from filinglens.evals.judge import Judge
            from filinglens.risk.classify import Classifier

            api = (
                anthropic.Anthropic(api_key=settings.anthropic_api_key)
                if settings.anthropic_api_key
                else anthropic.Anthropic()
            )
            ctx.classifier = Classifier(
                api, model=settings.classifier_model, effort=settings.classifier_effort
            )
            ctx.judge = Judge(api, model=settings.judge_model, effort=settings.judge_effort)

            async def agent(scenario: str, value: str) -> Any:
                run = await run_scenario(scenario, value, settings=settings, project_root=Path.cwd())
                _log_run(database_url, run)
                return run

            ctx.agent = agent
            ctx.agent_plan = AGENT_PLAN
        report = anyio.run(functools.partial(run_suite, suite, ctx))

    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = report_dir / f"{suite}-{stamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    if database_url is not None:
        from filinglens.db import repository as repo
        from filinglens.db.session import make_engine, session_scope

        with session_scope(make_engine(database_url)) as session:
            run_id = repo.insert_eval_report(session, report)
        click.echo(f"stored eval run {run_id}", err=True)
    click.echo(_render(report))
    click.echo(f"report: {path}")
    if not report.passed:
        sys.exit(1)


@eval_group.command("snapshot")
@click.option("--tickers", default="AAPL,JPM,COST,WMT,XOM", show_default=True)
def eval_snapshot(tickers: str) -> None:
    """Write the pipeline's current values as UNVERIFIED golden rows (verified rows are kept)."""
    from filinglens.cadence import compute_cadence, years_before
    from filinglens.cli import _client
    from filinglens.financials import compute_financials
    from filinglens.pipeline import company_filings, load_ticker_index

    universe = load_universe(COMPANIES)
    today = date.today()
    fin_new: list[golden.FinancialGolden] = []
    cad_new: list[golden.CadenceGolden] = []
    with _client() as client:
        index = load_ticker_index(client)
        for ticker in [t.strip().upper() for t in tickers.split(",") if t.strip()]:
            entry = index.resolve(ticker)
            report = compute_financials(client.company_facts(entry.cik), cik=entry.cik, name=entry.title)
            for series in (report.revenue, report.net_income):
                for q in series.latest_two:
                    fin_new.append(
                        golden.FinancialGolden(
                            case_id=f"{entry.ticker}-{series.metric.value}-FY{q.fiscal_year}Q{q.fiscal_quarter}",
                            ticker=entry.ticker,
                            metric=series.metric.value,
                            fiscal_year=q.fiscal_year or 0,
                            fiscal_quarter=q.fiscal_quarter or 0,
                            period_end=q.period_end,
                            value_usd=str(q.value),
                            accession=q.accession,
                            note="derived: FY - 9M" if q.derived else None,
                        )
                    )
            subs, filings, _ = company_filings(
                client, entry.cik, years_before(today, 3), universe.predecessors_for(ticker)
            )
            cadence = compute_cadence(
                cik=entry.cik,
                name=entry.title,
                fiscal_year_end=subs.get("fiscalYearEnd"),
                filings=filings,
                as_of=today,
            )
            cad_new += [
                golden.CadenceGolden(
                    case_id=f"{entry.ticker}-{f.form}-{today}",
                    ticker=entry.ticker,
                    as_of=today,
                    form=f.form,
                    count=f.count,
                )
                for f in cadence.forms
            ]
    for model, name, new in (
        (golden.FinancialGolden, golden.FINANCIALS, fin_new),
        (golden.CadenceGolden, golden.CADENCE, cad_new),
    ):
        existing = golden.load(model, name)
        verified = {r.case_id for r in existing if r.verified_by}  # type: ignore[attr-defined]
        merged = golden.upsert(existing, [r for r in new if r.case_id not in verified])
        path = golden.save(merged, name)
        click.echo(f"{path}: {len(merged)} rows ({len(verified)} verified kept)")
    click.echo("Next: `filinglens eval label financials` and `filinglens eval label cadence` to verify them.")


@eval_group.group("label")
def label() -> None:
    """Hand-label golden data (interactive)."""


LABELER = click.option("--labeler", prompt="Your name (stored with each label)", help="Who is labelling.")


def _edgar_link(ticker: str, accession: str) -> str:
    """A link for a person to open (EDGAR's filing list for the company), plus the accession.

    The accession prefix is often a filing agent's CIK, so it can't build an Archive path."""
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-"
    return f"{url}  (accession {accession})"


@label.command("financials")
@LABELER
def label_financials(labeler: str) -> None:
    """Verify snapshot revenue / net income values against the filings."""
    rows = golden.load(golden.FinancialGolden, golden.FINANCIALS)
    todo = [r for r in rows if not r.verified_by]
    click.echo(
        f"{len(todo)} unverified rows. Answer y (matches the filing), a corrected value, "
        "s (skip) or q (quit)."
    )
    for r in todo:
        click.echo(
            f"\n{r.ticker} {r.metric} FY{r.fiscal_year} Q{r.fiscal_quarter} "
            f"ending {r.period_end}: ${r.value_usd}"
            f"  {r.note or ''}\n  filing: {_edgar_link(r.ticker, r.accession)}"
        )
        answer = click.prompt("  matches?", default="s").strip()
        if answer.lower() == "q":
            break
        if answer.lower() == "s":
            continue
        if answer.lower() != "y":
            r.value_usd = answer.replace(",", "").replace("$", "")
            r.note = f"corrected by {labeler}"
        r.verified_by, r.verified_at = labeler, date.today()
    golden.save(rows, golden.FINANCIALS)
    click.echo(f"saved; {sum(bool(r.verified_by) for r in rows)} verified rows")


@label.command("cadence")
@LABELER
def label_cadence(labeler: str) -> None:
    """Verify snapshot filing counts (check EDGAR's filing index for the company)."""
    rows = golden.load(golden.CadenceGolden, golden.CADENCE)
    todo = [r for r in rows if not r.verified_by]
    click.echo(f"{len(todo)} unverified rows. Answer y, a corrected count, s (skip) or q (quit).")
    for r in todo:
        click.echo(f"\n{r.ticker}: {r.count} x {r.form} filed in the 3 years to {r.as_of}")
        answer = click.prompt("  correct?", default="s").strip()
        if answer.lower() == "q":
            break
        if answer.lower() == "s":
            continue
        if answer.lower() != "y":
            r.count = int(answer)
            r.note = f"corrected by {labeler}"
        r.verified_by, r.verified_at = labeler, date.today()
    golden.save(rows, golden.CADENCE)
    click.echo(f"saved; {sum(bool(r.verified_by) for r in rows)} verified rows")


@label.command("risk")
@click.argument("ticker")
@LABELER
@click.option("--max", "max_factors", default=50, show_default=True)
def label_risk(ticker: str, labeler: str, max_factors: int) -> None:
    """Label the categories of TICKER's latest 10-K risk factors."""
    from filinglens.cadence import years_before
    from filinglens.cli import _client
    from filinglens.pipeline import company_filings, load_ticker_index
    from filinglens.risk.classify import CATEGORY_GUIDE, Category
    from filinglens.risk.documents import extract_risk_section, recent_annual_reports

    universe = load_universe(COMPANIES)
    with _client() as client:
        entry = load_ticker_index(client).resolve(ticker)
        _, filings, _ = company_filings(
            client, entry.cik, years_before(date.today(), 2), universe.predecessors_for(entry.ticker)
        )
        report = recent_annual_reports(filings, entry.cik, client.company_facts(entry.cik), count=1)[0]
        extracted = extract_risk_section(client, report)
    rows = golden.load(golden.RiskLabel, golden.RISK_LABELS)
    done = {r.case_id for r in rows}
    valid = {c.value for c in Category}
    click.echo("Categories: " + "; ".join(f"{c.value} ({g})" for c, g in CATEGORY_GUIDE.items()))
    click.echo("Enter 1-3 comma-separated categories (most important first), s to skip, q to quit.")
    for f in extracted.factors[:max_factors]:
        key = golden.heading_key(f.heading, f.body)
        case_id = f"{entry.ticker}-FY{report.fiscal_year}-{key}"
        if case_id in done:
            continue
        click.echo(f"\n[{f.ordinal}] {f.heading or '(no heading)'}\n    {' '.join(f.body.split()[:70])}...")
        while True:
            answer = click.prompt("  categories", default="s").strip().lower()
            if answer in ("s", "q"):
                break
            cats = [c.strip() for c in answer.split(",") if c.strip()]
            if 1 <= len(cats) <= 3 and set(cats) <= valid:
                rows.append(
                    golden.RiskLabel(
                        case_id=case_id,
                        ticker=entry.ticker,
                        accession=report.accession,
                        fiscal_year=report.fiscal_year,
                        heading_key=key,
                        heading=f.heading,
                        categories=cats,
                        labeler=labeler,
                        labeled_at=date.today(),
                    )
                )
                break
            click.echo(f"  use 1-3 of: {', '.join(sorted(valid))}")
        if answer == "q":
            break
        golden.save(rows, golden.RISK_LABELS)  # save as you go
    golden.save(rows, golden.RISK_LABELS)
    click.echo(f"saved {sum(r.ticker == entry.ticker for r in rows)} labels for {entry.ticker}")


@label.command("coherence")
@click.argument("report_path", type=click.Path(exists=True, path_type=Path))
@LABELER
def label_coherence(report_path: Path, labeler: str) -> None:
    """Score the agent outputs in a full-suite REPORT for coherence (1-5), blind to the judge."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = golden.load(golden.CoherenceLabel, golden.COHERENCE_LABELS)
    done = {(r.case_id, r.output_sha) for r in rows}
    cases = [c for c in report["cases"] if c["metric"] == "coherence" and c["details"].get("output")]
    click.echo("5 clear and consistent ... 1 incoherent. s to skip, q to quit.")
    for c in cases:
        sha = c["details"]["output_sha"]
        if (c["case_id"], sha) in done:
            continue
        click.echo(f"\n=== {c['case_id']} ===\n{json.dumps(c['details']['output'], indent=2)[:6000]}")
        answer = click.prompt("  score", default="s").strip().lower()
        if answer == "q":
            break
        if answer not in {"1", "2", "3", "4", "5"}:
            continue
        rows.append(
            golden.CoherenceLabel(
                case_id=c["case_id"],
                scenario=c["case_id"].split("-", 1)[0],
                output_sha=sha,
                score=int(answer),
                judge_score=int(c["score"]) if c.get("score") is not None else None,
                labeler=labeler,
                labeled_at=date.today(),
            )
        )
    golden.save(rows, golden.COHERENCE_LABELS)
    click.echo(f"saved; {len(rows)} coherence labels")
