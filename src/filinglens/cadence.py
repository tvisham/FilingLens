"""Deterministic filing-cadence analysis from EDGAR submissions JSON.

The submissions endpoint returns columnar arrays: `filings.recent` holds the latest
filings (at least one year, up to 1,000), and `filings.files[]` points to older pages with
the same columnar shape. Heavy 8-K filers overflow `recent` well within three years, so
pages that overlap the window are always merged in.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel

TRACKED_FORMS: tuple[str, ...] = ("10-K", "10-Q", "8-K", "DEF 14A")
QUARTERLY_REPORTS_PER_FISCAL_YEAR = 3  # Q4 is covered by the 10-K


class Filing(BaseModel):
    accession: str
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str | None
    # CIK whose submissions listed this filing. Differs from the company's current CIK for
    # filings made by a predecessor registrant, and EDGAR Archive URLs are per-filer.
    filer_cik: str | None = None
    items: list[str] = []  # 8-K item codes, e.g. ["2.02", "9.01"]
    description: str | None = None  # SEC's primaryDocDescription

    @property
    def is_amendment(self) -> bool:
        return self.form.endswith("/A")

    @property
    def base_form(self) -> str:
        return self.form.removesuffix("/A")


class FormSummary(BaseModel):
    form: str
    count: int
    amendments: int
    dates: list[date]
    median_days_after_period_end: float | None


class FiscalPeriodCheck(BaseModel):
    """Quarterly reports filed between two consecutive annual reports."""

    period_start_exclusive: date | None  # previous 10-K period end, None if outside window
    period_end: date | None  # this 10-K period end, None for the in-progress year
    ten_q_count: int
    expected: int
    complete: bool
    in_progress: bool = False


class CadenceReport(BaseModel):
    cik: str
    name: str
    fiscal_year_end: str | None  # MMDD as published by SEC, e.g. "0927"
    window_start: date
    window_end: date
    forms: list[FormSummary]
    fiscal_periods: list[FiscalPeriodCheck]


# -- parsing -----------------------------------------------------------------------------


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value)[:10])


def filing_url(cik: str, accession: str, primary_document: str | None) -> str:
    """EDGAR Archive URL. Archive paths are per filer, so pass the CIK that filed it."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    return f"{base}/{primary_document}" if primary_document else f"{base}/"


def parse_filing_block(block: Mapping[str, Sequence[Any]], *, filer_cik: str | None = None) -> list[Filing]:
    """Turn one columnar block (`filings.recent` or a page file) into `Filing` rows."""
    accessions = block.get("accessionNumber", [])
    n = len(accessions)
    for column in ("form", "filingDate"):
        if len(block.get(column, [])) != n:
            raise ValueError(f"column {column!r} length does not match accessionNumber")
    report_dates = block.get("reportDate", [None] * n)
    docs = block.get("primaryDocument", [None] * n)
    items = block.get("items", [""] * n)
    descriptions = block.get("primaryDocDescription", [""] * n)
    out: list[Filing] = []
    for i in range(n):
        filing_date = _parse_date(block["filingDate"][i])
        if filing_date is None:
            continue
        out.append(
            Filing(
                accession=str(accessions[i]),
                form=str(block["form"][i]),
                filing_date=filing_date,
                report_date=_parse_date(report_dates[i]) if i < len(report_dates) else None,
                primary_document=(str(docs[i]) or None) if i < len(docs) and docs[i] else None,
                filer_cik=filer_cik,
                items=[s.strip() for s in str(items[i] or "").split(",") if s.strip()]
                if i < len(items)
                else [],
                description=(str(descriptions[i]).strip() or None)
                if i < len(descriptions) and descriptions[i]
                else None,
            )
        )
    return out


def pages_overlapping(submissions: Mapping[str, Any], window_start: date) -> list[str]:
    """Names of older submission pages whose date range reaches into the window."""
    names: list[str] = []
    for page in submissions.get("filings", {}).get("files", []):
        filing_to = _parse_date(page.get("filingTo"))
        if filing_to is None or filing_to >= window_start:
            names.append(str(page["name"]))
    return names


def merge_filings(blocks: Iterable[list[Filing]]) -> list[Filing]:
    """Merge filing lists, de-duplicating by accession number, sorted by filing date.

    The first block that lists an accession wins, so pass the current registrant's blocks
    before any predecessor's.
    """
    by_accession: dict[str, Filing] = {}
    for block in blocks:
        for f in block:
            by_accession.setdefault(f.accession, f)
    return sorted(by_accession.values(), key=lambda f: (f.filing_date, f.accession))


def years_before(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # Feb 29 -> Feb 28
        return d.replace(year=d.year - years, day=28)


# -- analysis ----------------------------------------------------------------------------


def _summarise_form(form: str, filings: list[Filing]) -> FormSummary:
    originals = [f for f in filings if f.form == form]
    amendments = [f for f in filings if f.form == f"{form}/A"]
    lags = [
        (f.filing_date - f.report_date).days
        for f in originals
        if f.report_date is not None and form in ("10-K", "10-Q")
    ]
    return FormSummary(
        form=form,
        count=len(originals),
        amendments=len(amendments),
        dates=sorted(f.filing_date for f in originals),
        median_days_after_period_end=float(statistics.median(lags)) if lags else None,
    )


def _fiscal_period_checks(filings: list[Filing]) -> list[FiscalPeriodCheck]:
    annual_ends = sorted({f.report_date for f in filings if f.form == "10-K" and f.report_date is not None})
    quarter_ends = sorted(f.report_date for f in filings if f.form == "10-Q" and f.report_date is not None)
    checks: list[FiscalPeriodCheck] = []
    previous: date | None = None
    for end in annual_ends:
        if previous is not None:  # the first 10-K's year may start before the window
            count = sum(1 for q in quarter_ends if previous < q < end)
            checks.append(
                FiscalPeriodCheck(
                    period_start_exclusive=previous,
                    period_end=end,
                    ten_q_count=count,
                    expected=QUARTERLY_REPORTS_PER_FISCAL_YEAR,
                    complete=count == QUARTERLY_REPORTS_PER_FISCAL_YEAR,
                )
            )
        previous = end
    if previous is not None:
        in_progress = sum(1 for q in quarter_ends if q > previous)
        checks.append(
            FiscalPeriodCheck(
                period_start_exclusive=previous,
                period_end=None,
                ten_q_count=in_progress,
                expected=QUARTERLY_REPORTS_PER_FISCAL_YEAR,
                complete=in_progress == QUARTERLY_REPORTS_PER_FISCAL_YEAR,
                in_progress=True,
            )
        )
    return checks


def compute_cadence(
    *,
    cik: str,
    name: str,
    fiscal_year_end: str | None,
    filings: list[Filing],
    as_of: date,
    years: int = 3,
) -> CadenceReport:
    window_start = years_before(as_of, years)
    in_window = [f for f in filings if window_start <= f.filing_date <= as_of]
    return CadenceReport(
        cik=cik,
        name=name,
        fiscal_year_end=fiscal_year_end,
        window_start=window_start,
        window_end=as_of,
        forms=[_summarise_form(form, in_window) for form in TRACKED_FORMS],
        fiscal_periods=_fiscal_period_checks(in_window),
    )
