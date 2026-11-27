"""Post-run guardrails and the decision to publish or queue for human review.

A run is published only if it finished cleanly and every check passed. Anything else goes
to the review queue with its reasons: `block` issues (wrong numbers, investment advice,
invented citations, invalid output) must be fixed or rejected by a person; `review` issues
(an unverifiable dollar amount, a low-confidence extraction) need a look before release.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from filinglens.agent.records import AgentRunResult
from filinglens.agent.scenarios import DISCLAIMER
from filinglens.edgar.client import SecClient, SecNotFoundError
from filinglens.edgar.tickers import TickerIndex, UnknownTickerError
from filinglens.financials import compute_financials
from filinglens.guardrails.checks import (
    GuardrailResult,
    Issue,
    KnownFigure,
    check_numbers,
    check_policy,
    enforce_disclaimer,
    strings,
    tool_flags,
)

_STATUS_ISSUES = {
    "invalid_output": "output failed schema validation after a repair attempt",
    "unsupported_citation": "cited chunk ids that no tool returned in this run",
    "disallowed_tool": "called a tool outside the allowlist",
    "budget_exceeded": "stopped at the USD budget",
    "max_turns": "stopped at the turn limit",
    "error": "the run failed",
}


def tickers_in(run: AgentRunResult) -> list[str]:
    """Tickers the run is about: its input plus any `ticker` fields in its output."""
    found = {str(run.input["ticker"]).upper()} if "ticker" in run.input else set()
    for path, text in strings(run.output or {}):
        if path == "ticker" or path.endswith(".ticker"):
            found.add(text.upper())
    return sorted(found)


def known_figures(client: SecClient, index: TickerIndex, tickers: Iterable[str]) -> list[KnownFigure]:
    """Every quarterly revenue / net income value the deterministic pipeline knows."""
    out: list[KnownFigure] = []
    for ticker in tickers:
        try:
            entry = index.resolve(ticker)
            facts = client.company_facts(entry.cik)
        except (UnknownTickerError, SecNotFoundError):
            continue
        report = compute_financials(facts, cik=entry.cik, name=entry.title)
        for series in (report.revenue, report.net_income):
            out += [
                KnownFigure(entry.ticker, series.metric.value, q.period_end.isoformat(), q.value, q.accession)
                for q in series.quarters
            ]
    return out


def guard(run: AgentRunResult, known: Sequence[KnownFigure]) -> GuardrailResult:
    """Check a finished run. Mutates `run.output` only to add the standard disclaimer."""
    result = GuardrailResult()
    if run.status != "ok":
        result.issues.append(
            Issue("status", "block", _STATUS_ISSUES.get(run.status, run.status) + f": {run.error}")
        )
    if run.output:
        result.fixes += enforce_disclaimer(run.output, DISCLAIMER)
        tool_text = " ".join(str(p) for p in run.tool_payloads)
        result.issues += check_numbers(run.output, list(known), tool_text)
        result.issues += check_policy(run.output)
    result.issues += tool_flags(run.tool_payloads)
    run.guardrails = result.to_dict()
    return result
