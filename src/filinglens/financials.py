"""Deterministic quarterly revenue and net income from XBRL company facts.

Key rules (each one is covered by unit tests):

* Periods are classified by their start/end dates and length, never by the fact's
  `fy`/`fp` fields. Those describe the *filing* a fact came from, so a 10-K's prior-year
  comparatives carry the new year's `fy`.
* Quarter = 80-120 days (12-, 13-, 14- and 16-week quarters), 9-month YTD = 245-290 days,
  annual = 350-380 days (52/53-week years).
* For the same concept and period, the value comes from the most recently filed fact
  (restatements win) and its accession number is stored.
* Fiscal year labels come from the annual period's original 10-K, and quarter numbers
  come from the quarter's position inside that annual period.
* Q4 is rarely reported on its own. It is derived as FY minus 9-month YTD, falling back
  to FY minus (Q1 + Q2 + Q3); derived values are flagged with both source accessions.
* Revenue concept: the first concept in the fallback order that covers the most recent
  quarter wins, so a tag the company stopped using years ago is never chosen.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

REVENUE_CONCEPTS: tuple[str, ...] = (
    # Banks report headline revenue net of interest expense; non-banks don't use this tag.
    "RevenuesNetOfInterestExpense",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
NET_INCOME_CONCEPTS: tuple[str, ...] = (
    "NetIncomeLoss",  # attributable to the parent: preferred
    "ProfitLoss",  # includes noncontrolling interests
    "NetIncomeLossAvailableToCommonStockholdersBasic",
)
ALLOWED_FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "10-QT"})

QUARTER_DAYS = range(80, 121)
YTD9_DAYS = range(245, 291)
ANNUAL_DAYS = range(350, 381)
DATE_TOLERANCE_DAYS = 1
WARNING_LOOKBACK_YEARS = 3  # matches the cadence window


class Metric(StrEnum):
    REVENUE = "revenue"
    NET_INCOME = "net_income"


class PeriodType(StrEnum):
    QUARTER = "quarter"
    YTD9 = "ytd9"
    ANNUAL = "annual"
    OTHER = "other"


def classify_duration(start: date, end: date) -> PeriodType:
    days = (end - start).days
    if days in QUARTER_DAYS:
        return PeriodType.QUARTER
    if days in YTD9_DAYS:
        return PeriodType.YTD9
    if days in ANNUAL_DAYS:
        return PeriodType.ANNUAL
    return PeriodType.OTHER


class FactRow(BaseModel):
    """One raw XBRL duration fact as it appears in companyfacts."""

    concept: str
    start: date
    end: date
    value: Decimal
    unit: str
    accession: str
    fy: int | None
    fp: str | None
    form: str
    filed: date


class Period(BaseModel):
    """A single deduplicated (or derived) value for one concept and one period."""

    concept: str
    period_start: date
    period_end: date
    period_type: PeriodType
    value: Decimal
    unit: str
    accession: str
    form: str
    filed: date
    label_fy: int | None = None  # fy/fp from the earliest filing that reported this period
    label_fp: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    derived: bool = False
    derivation: str | None = None
    source_accessions: list[str] = Field(default_factory=list)

    @property
    def duration_days(self) -> int:
        return (self.period_end - self.period_start).days


class MetricResult(BaseModel):
    metric: Metric
    concept: str | None
    concepts_considered: list[str]
    quarters: list[Period]  # all fiscal quarters found, oldest first
    latest_two: list[Period]  # most recent two completed quarters, oldest first
    warnings: list[str] = Field(default_factory=list)


class FinancialsReport(BaseModel):
    cik: str
    name: str
    revenue: MetricResult
    net_income: MetricResult


# -- extraction --------------------------------------------------------------------------


def extract_rows(
    companyfacts: Mapping[str, Any],
    concept: str,
    *,
    taxonomy: str = "us-gaap",
    unit: str = "USD",
) -> list[FactRow]:
    """Duration facts for one concept from allowed forms, in the given unit."""
    node = companyfacts.get("facts", {}).get(taxonomy, {}).get(concept)
    if not node:
        return []
    rows: list[FactRow] = []
    for raw in node.get("units", {}).get(unit, []):
        if "start" not in raw or raw.get("form") not in ALLOWED_FORMS:
            continue
        rows.append(
            FactRow(
                concept=concept,
                start=date.fromisoformat(raw["start"]),
                end=date.fromisoformat(raw["end"]),
                value=Decimal(str(raw["val"])),
                unit=unit,
                accession=str(raw["accn"]),
                fy=int(raw["fy"]) if raw.get("fy") is not None else None,
                fp=str(raw["fp"]) if raw.get("fp") else None,
                form=str(raw["form"]),
                filed=date.fromisoformat(raw["filed"]),
            )
        )
    return rows


def dedupe(rows: Sequence[FactRow]) -> list[Period]:
    """Collapse repeated facts for the same period.

    Value and accession come from the latest filing (restatements win). The fiscal label
    comes from the earliest filing, which is the one where this period was the current one.
    """
    groups: dict[tuple[date, date], list[FactRow]] = defaultdict(list)
    for row in rows:
        groups[(row.start, row.end)].append(row)
    periods: list[Period] = []
    for (start, end), group in groups.items():
        latest = max(group, key=lambda r: (r.filed, r.accession))
        earliest = min(group, key=lambda r: (r.filed, r.accession))
        periods.append(
            Period(
                concept=latest.concept,
                period_start=start,
                period_end=end,
                period_type=classify_duration(start, end),
                value=latest.value,
                unit=latest.unit,
                accession=latest.accession,
                form=latest.form,
                filed=latest.filed,
                label_fy=earliest.fy,
                label_fp=earliest.fp,
                source_accessions=[latest.accession],
            )
        )
    return sorted(periods, key=lambda p: (p.period_end, p.period_start))


# -- fiscal quarters ---------------------------------------------------------------------


def _close(a: date, b: date, tol: int = DATE_TOLERANCE_DAYS) -> bool:
    return abs((a - b).days) <= tol


def _follows(prev_end: date, next_start: date) -> bool:
    return 0 <= (next_start - prev_end).days <= DATE_TOLERANCE_DAYS


def _derive_q4(annual: Period, ytd9s: list[Period], quarters: list[Period]) -> Period | None:
    # 1) FY minus 9-month YTD
    for ytd in sorted(ytd9s, key=lambda p: p.filed, reverse=True):
        if _close(ytd.period_start, annual.period_start) and ytd.period_end < annual.period_end:
            start = ytd.period_end + timedelta(days=1)
            if classify_duration(start, annual.period_end) is PeriodType.QUARTER:
                return annual.model_copy(
                    update={
                        "period_start": start,
                        "period_type": PeriodType.QUARTER,
                        "value": annual.value - ytd.value,
                        "derived": True,
                        "derivation": "FY - 9M YTD",
                        "source_accessions": [annual.accession, ytd.accession],
                        "label_fp": "Q4",
                    }
                )
    # 2) FY minus (Q1 + Q2 + Q3), requiring three contiguous quarters from the FY start
    inside = sorted(
        (
            q
            for q in quarters
            if annual.period_start - timedelta(days=DATE_TOLERANCE_DAYS) <= q.period_start
            and q.period_end < annual.period_end
        ),
        key=lambda q: q.period_start,
    )
    if len(inside) >= 3:
        q1, q2, q3 = inside[:3]
        if (
            _close(q1.period_start, annual.period_start)
            and _follows(q1.period_end, q2.period_start)
            and _follows(q2.period_end, q3.period_start)
        ):
            start = q3.period_end + timedelta(days=1)
            if classify_duration(start, annual.period_end) is PeriodType.QUARTER:
                return annual.model_copy(
                    update={
                        "period_start": start,
                        "period_type": PeriodType.QUARTER,
                        "value": annual.value - q1.value - q2.value - q3.value,
                        "derived": True,
                        "derivation": "FY - (Q1 + Q2 + Q3)",
                        "source_accessions": [annual.accession, q1.accession, q2.accession, q3.accession],
                        "label_fp": "Q4",
                    }
                )
    return None


_FP_TO_QUARTER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _collapse_near_duplicates(quarters: list[Period]) -> list[Period]:
    """One quarter per period. A later filing can restate a quarter with a start date a day
    off (Microsoft's FY2018 10-K gives Q1 FY2017 as 2016-07-02 to 2016-09-30; the original
    10-Q said 2016-07-01). Exact-date de-duplication keeps both, which would make five
    "quarters" in the year; here the most recently filed version wins, as for restatements."""
    kept: list[Period] = []
    for q in sorted(quarters, key=lambda p: p.filed, reverse=True):
        if not any(
            _close(q.period_start, k.period_start) and _close(q.period_end, k.period_end) for k in kept
        ):
            kept.append(q)
    return sorted(kept, key=lambda p: p.period_end)


def fiscal_quarters(
    periods: Sequence[Period], *, warn_after: date | None = None
) -> tuple[list[Period], list[str]]:
    """All fiscal quarters (reported + derived Q4), labelled with fiscal year and quarter.

    A missing Q4 is only warned about for fiscal years ending on or after `warn_after`, so old
    gaps (e.g. 2008-era filings before quarterly XBRL) don't drown out recent ones.
    """
    warnings: list[str] = []
    quarters = _collapse_near_duplicates([p for p in periods if p.period_type is PeriodType.QUARTER])
    ytd9s = [p for p in periods if p.period_type is PeriodType.YTD9]
    annuals = sorted((p for p in periods if p.period_type is PeriodType.ANNUAL), key=lambda p: p.period_end)

    for annual in annuals:
        has_q4 = any(
            _close(q.period_end, annual.period_end) and q.period_start > annual.period_start for q in quarters
        )
        if has_q4:
            continue
        q4 = _derive_q4(annual, ytd9s, quarters)
        if q4 is None:
            if warn_after is not None and annual.period_end < warn_after:
                continue
            warnings.append(
                f"{annual.concept}: could not derive Q4 for fiscal year ending {annual.period_end}"
            )
        else:
            quarters.append(q4)

    labelled: list[Period] = []
    for q in sorted(quarters, key=lambda p: p.period_end):
        container = next(
            (
                a
                for a in annuals
                if a.period_start - timedelta(days=DATE_TOLERANCE_DAYS) <= q.period_start
                and q.period_end <= a.period_end + timedelta(days=DATE_TOLERANCE_DAYS)
            ),
            None,
        )
        if container is not None:
            siblings = sorted(
                (
                    s
                    for s in quarters
                    if container.period_start - timedelta(days=DATE_TOLERANCE_DAYS) <= s.period_start
                    and s.period_end <= container.period_end + timedelta(days=DATE_TOLERANCE_DAYS)
                ),
                key=lambda s: s.period_end,
            )
            index = next(i for i, s in enumerate(siblings, 1) if s is q)
            fiscal_year = (
                container.label_fy if container.label_fy is not None else (container.period_end.year)
            )
            if index > 4:  # should be impossible after collapsing; never emit a "Q5"
                warnings.append(
                    f"{q.concept}: more than four quarters in the fiscal year ending {container.period_end}; "
                    f"{q.period_start} to {q.period_end} left unlabelled"
                )
                labelled.append(q.model_copy(update={"fiscal_year": fiscal_year, "fiscal_quarter": None}))
                continue
            labelled.append(q.model_copy(update={"fiscal_year": fiscal_year, "fiscal_quarter": index}))
        else:
            # Current fiscal year, no 10-K yet: trust the original filing's label.
            labelled.append(
                q.model_copy(
                    update={
                        "fiscal_year": q.label_fy,
                        "fiscal_quarter": _FP_TO_QUARTER.get(q.label_fp or ""),
                    }
                )
            )
    return labelled, warnings


# -- top level ---------------------------------------------------------------------------


def metric_result(companyfacts: Mapping[str, Any], metric: Metric, concepts: Sequence[str]) -> MetricResult:
    per_concept: dict[str, list[Period]] = {}
    warnings: dict[str, list[str]] = {}
    for concept in concepts:
        rows = extract_rows(companyfacts, concept)
        if not rows:
            continue
        latest = max(r.end for r in rows)
        cutoff = latest - timedelta(days=round(365.25 * WARNING_LOOKBACK_YEARS))
        quarters, warns = fiscal_quarters(dedupe(rows), warn_after=cutoff)
        if quarters:
            per_concept[concept] = quarters
            warnings[concept] = warns

    considered = [c for c in concepts if c in per_concept]
    if not per_concept:
        return MetricResult(
            metric=metric,
            concept=None,
            concepts_considered=[],
            quarters=[],
            latest_two=[],
            warnings=[f"no quarterly USD data for any of: {', '.join(concepts)}"],
        )

    latest_end = max(q.period_end for qs in per_concept.values() for q in qs)
    chosen = next(
        c
        for c in concepts
        if c in per_concept and any(_close(q.period_end, latest_end) for q in per_concept[c])
    )
    quarters = per_concept[chosen]
    return MetricResult(
        metric=metric,
        concept=chosen,
        concepts_considered=considered,
        quarters=quarters,
        latest_two=quarters[-2:],
        warnings=warnings[chosen],
    )


def compute_financials(companyfacts: Mapping[str, Any], *, cik: str, name: str) -> FinancialsReport:
    return FinancialsReport(
        cik=cik,
        name=name,
        revenue=metric_result(companyfacts, Metric.REVENUE, REVENUE_CONCEPTS),
        net_income=metric_result(companyfacts, Metric.NET_INCOME, NET_INCOME_CONCEPTS),
    )
