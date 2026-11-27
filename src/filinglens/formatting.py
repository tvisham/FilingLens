"""Plain-text rendering of reports for the CLI."""

from __future__ import annotations

from decimal import Decimal

from filinglens.cadence import CadenceReport
from filinglens.financials import FinancialsReport, MetricResult


def money(value: Decimal) -> str:
    """Human-readable USD: $1.23B, $456.7M, $12.3K."""
    sign = "-" if value < 0 else ""
    v = abs(value)
    for threshold, suffix in ((Decimal(10**9), "B"), (Decimal(10**6), "M"), (Decimal(10**3), "K")):
        if v >= threshold:
            return f"{sign}${v / threshold:,.2f}{suffix}"
    return f"{sign}${v:,.0f}"


def render_cadence(report: CadenceReport) -> str:
    lines = [
        f"{report.name} (CIK {report.cik})",
        f"Fiscal year end (MMDD): {report.fiscal_year_end or 'unknown'}",
        f"Window: {report.window_start} to {report.window_end}",
        "",
        f"{'Form':<9}{'Count':>6}{'Amend.':>8}{'Median lag (days)':>19}  Latest",
    ]
    for f in report.forms:
        lag = f"{f.median_days_after_period_end:.0f}" if f.median_days_after_period_end else "-"
        latest = f.dates[-1].isoformat() if f.dates else "-"
        lines.append(f"{f.form:<9}{f.count:>6}{f.amendments:>8}{lag:>19}  {latest}")
    lines += ["", "Fiscal years (10-Qs between consecutive 10-Ks, expected 3):"]
    for p in report.fiscal_periods:
        end = p.period_end.isoformat() if p.period_end else "in progress"
        status = "ok" if p.complete else ("in progress" if p.in_progress else "MISSING")
        lines.append(f"  after {p.period_start_exclusive} -> {end}: {p.ten_q_count}/{p.expected} {status}")
    return "\n".join(lines)


def _render_metric(title: str, result: MetricResult) -> list[str]:
    lines = [f"{title}  [concept: {result.concept or 'none'}]"]
    for q in result.latest_two:
        label = f"FY{q.fiscal_year} Q{q.fiscal_quarter}" if q.fiscal_quarter else f"FY{q.fiscal_year}"
        flag = f"  (derived: {q.derivation})" if q.derived else ""
        lines.append(
            f"  {label:<10} {q.period_start} to {q.period_end}  {money(q.value):>12}"
            f"  accn {q.accession}{flag}"
        )
    lines += [f"  ! {w}" for w in result.warnings]
    return lines


def render_financials(report: FinancialsReport) -> str:
    lines = [f"{report.name} (CIK {report.cik}) - last two fiscal quarters", ""]
    lines += _render_metric("Revenue", report.revenue)
    lines.append("")
    lines += _render_metric("Net income", report.net_income)
    return "\n".join(lines)
