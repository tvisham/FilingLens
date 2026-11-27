"""Run an eval suite and build its report.

smoke  no LLM, a few companies: gates every pull request.
full   smoke over the whole universe + every LLM eval: nightly or on demand (costs money,
       capped by EVAL_MAX_COST_USD; the agent part also needs the risk database).

Every metric is compared with `evals/thresholds.yaml`. A metric with nothing to measure
(e.g. no human labels yet) is reported as skipped and does not fail the gate.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from filinglens.config import Universe
from filinglens.edgar.client import SecClient
from filinglens.edgar.tickers import TickerIndex
from filinglens.evals import deterministic as det
from filinglens.evals import golden, silver
from filinglens.evals import llm_evals as llm
from filinglens.evals.judge import Judge
from filinglens.evals.results import CaseResult, EvalReport, MetricResult, judge_metric, share_passed
from filinglens.risk.classify import Classifier

SMOKE_TICKERS = ("AAPL", "JPM", "COST", "WMT", "XOM")


@dataclass
class EvalContext:
    client: SecClient
    index: TickerIndex
    universe: Universe
    thresholds: dict[str, dict[str, float]]
    server_factory: Callable[[], Any]  # MCP server for the protocol-level guardrail checks
    classifier: Classifier | None = None
    judge: Judge | None = None
    agent: llm.AgentFn | None = None
    agent_plan: Sequence[tuple[str, str]] = ()
    smoke_tickers: Sequence[str] = SMOKE_TICKERS
    faithfulness_tickers: Sequence[str] = ("AAPL", "XOM", "JPM")
    faithfulness_per_ticker: int = 4
    # Free (no LLM) classifier scored against your risk labels in every suite; report-only.
    local_classifier: Any = None
    # Stored risk factors (DB); scores the stored classifications against the filers' own
    # section headings ("silver" labels, see evals/silver.py). Report-only.
    risk_store: Any = None
    golden_dir: Any = golden.GOLDEN_DIR
    log: Callable[[str], None] = field(default=lambda _m: None)

    @property
    def predecessors(self) -> dict[str, list[str]]:
        return {c.ticker: c.predecessor_ciks for c in self.universe.companies() if c.predecessor_ciks}


def git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    # Outside a repository, `git rev-parse HEAD` fails and echoes "HEAD" to stdout.
    return sha if out.returncode == 0 and len(sha) == 40 else None


def _share(name: str, cases: list[CaseResult], ctx: EvalContext) -> MetricResult:
    ran = [c for c in cases if c.passed is not None]
    failed = [c.case_id for c in ran if not c.passed]
    return judge_metric(name, share_passed(cases), ctx.thresholds, len(ran), failed=failed[:25])


async def run_suite(suite: str, ctx: EvalContext) -> EvalReport:
    if suite not in ("smoke", "full"):
        raise ValueError(f"unknown suite {suite!r}")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    cases: list[CaseResult] = []
    metrics: list[MetricResult] = []
    tickers = list(ctx.smoke_tickers) if suite == "smoke" else [c.ticker for c in ctx.universe.companies()]

    def add(name: str, new: list[CaseResult]) -> list[CaseResult]:
        cases.extend(new)
        metrics.append(_share(name, new, ctx))
        return new

    ctx.log("financial_xbrl_match")
    add("financial_xbrl_match", det.financial_xbrl_match(ctx.client, ctx.index, tickers))
    ctx.log("golden values")
    fin_rows = golden.load(golden.FinancialGolden, golden.FINANCIALS, ctx.golden_dir)
    cad_rows = golden.load(golden.CadenceGolden, golden.CADENCE, ctx.golden_dir)
    add("financial_golden_match", det.financial_golden_match(ctx.client, ctx.index, fin_rows))
    add("cadence_golden_match", det.cadence_golden_match(ctx.client, ctx.index, cad_rows, ctx.predecessors))
    metrics[-2].details["unverified_rows"] = sum(not r.verified_by for r in fin_rows)
    metrics[-1].details["unverified_rows"] = sum(not r.verified_by for r in cad_rows)
    ctx.log("extraction_confidence")
    add("extraction_confidence", det.extraction_confidence(ctx.client, ctx.index, tickers, ctx.predecessors))
    ctx.log("adversarial_tools")
    add("adversarial_tools", await det.adversarial_tools(ctx.server_factory()))
    ctx.log("guardrail_checks")
    add("guardrail_checks", det.guardrail_checks())

    risk_labels = golden.load(golden.RiskLabel, golden.RISK_LABELS, ctx.golden_dir)
    if risk_labels and ctx.local_classifier is not None:
        ctx.log("local classifier vs your risk labels")
        local_cases, local_summary = llm.risk_categories(
            ctx.client, ctx.index, risk_labels, ctx.local_classifier, ctx.predecessors, llm.LLMEvalState()
        )
        for c in local_cases:
            c.metric = "local_category"
        cases.extend(local_cases)
        n_local = sum(c.passed is not None for c in local_cases)
        local_micro = local_summary.get("micro", {})
        metrics.append(
            judge_metric(
                "local_category_f1_micro", local_micro.get("f1"), ctx.thresholds, n_local, **local_summary
            )
        )
        metrics.append(
            judge_metric("local_category_kappa", local_summary.get("kappa_primary"), ctx.thresholds, n_local)
        )

    if ctx.risk_store is not None:
        ctx.log("stored classifications vs filers' section headings (silver labels)")
        ciks = [ctx.index.resolve(t).cik for t in tickers]
        silver_cases, sv = silver.silver_eval(ctx.risk_store, ciks)
        cases.extend(silver_cases)
        worst = sorted(
            ((h, ok, n) for h, (ok, n) in sv.by_heading.items() if n >= 3), key=lambda x: x[1] / x[2]
        )[:8]
        metrics.append(
            judge_metric(
                "silver_category_agreement",
                sv.agreement_primary,
                ctx.thresholds,
                sv.labelled,
                agreement_any=sv.agreement_any,
                factors_classified=sv.total,
                factors_with_specific_heading=sv.labelled,
                worst_headings=[f"{h} ({ok}/{n})" for h, ok, n in worst],
                failed=[c.case_id for c in silver_cases if not c.passed][:25],
                note="silver: filer section headings, not human labels",
            )
        )

    cost = 0.0
    if suite == "full":
        if ctx.classifier is None or ctx.judge is None or ctx.agent is None:
            raise ValueError("the full suite needs a classifier, a judge and an agent")
        state = llm.LLMEvalState()

        ctx.log("risk categories")
        labels = golden.load(golden.RiskLabel, golden.RISK_LABELS, ctx.golden_dir)
        cat_cases, summary = llm.risk_categories(
            ctx.client, ctx.index, labels, ctx.classifier, ctx.predecessors, state
        )
        cases.extend(cat_cases)
        micro = summary.get("micro", {})
        n = sum(c.passed is not None for c in cat_cases)
        metrics.append(judge_metric("risk_category_f1_micro", micro.get("f1"), ctx.thresholds, n, **summary))
        metrics.append(judge_metric("risk_category_kappa", summary.get("kappa_primary"), ctx.thresholds, n))

        ctx.log("faithfulness")
        add(
            "faithfulness",
            llm.faithfulness(
                ctx.classifier,
                ctx.judge,
                ctx.client,
                ctx.index,
                ctx.faithfulness_tickers,
                ctx.faithfulness_per_ticker,
                ctx.predecessors,
                state,
            ),
        )

        ctx.log("agent runs")
        agent_cases, perf = await llm.agent_runs(ctx.agent, ctx.judge, ctx.agent_plan, state)
        cases.extend(agent_cases)
        success = [c for c in agent_cases if c.metric == "agent_success"]
        metrics.append(_share("agent_success_rate", success, ctx))
        scores = [c.score for c in agent_cases if c.metric == "coherence" and c.score is not None]
        metrics.append(
            judge_metric(
                "coherence_mean", sum(scores) / len(scores) if scores else None, ctx.thresholds, len(scores)
            )
        )
        for name, value in perf.items():
            metrics.append(judge_metric(name, value, ctx.thresholds, len(state.agent_runs)))
        coh_labels = golden.load(golden.CoherenceLabel, golden.COHERENCE_LABELS, ctx.golden_dir)
        rho, pairs = llm.judge_human_agreement(coh_labels)
        metrics.append(judge_metric("judge_human_spearman", rho, ctx.thresholds, pairs))

        ctx.log("adversarial llm")
        add("adversarial_llm", await llm.adversarial_llm(ctx.classifier, ctx.agent))

        agent_cost = sum(float(r.cost_usd or 0) for r in state.agent_runs)
        cost = float(ctx.classifier.spent_usd) + agent_cost
        cost += float(ctx.judge.usage.cost_usd)

    return EvalReport(
        suite=suite,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=git_sha(),
        models={
            "classifier": ctx.classifier.model if ctx.classifier else "",
            "judge": ctx.judge.model if ctx.judge else "",
        },
        metrics=metrics,
        cases=cases,
        cost_usd=round(cost, 6),
    )
