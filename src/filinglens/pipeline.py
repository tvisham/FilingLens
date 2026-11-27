"""Glue between the SEC client, the pure analysis modules, and the database."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from filinglens.cadence import (
    CadenceReport,
    Filing,
    compute_cadence,
    merge_filings,
    pages_overlapping,
    parse_filing_block,
    years_before,
)
from filinglens.edgar.client import SecNotFoundError
from filinglens.edgar.source import EdgarSource
from filinglens.edgar.tickers import TickerEntry, TickerIndex
from filinglens.financials import FinancialsReport, compute_financials

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


@dataclass
class CompanySnapshot:
    entry: TickerEntry
    sector: str
    submissions: dict[str, Any]
    filings: list[Filing]
    cadence: CadenceReport
    financials: FinancialsReport | None
    warnings: list[str] = field(default_factory=list)


def load_ticker_index(client: EdgarSource) -> TickerIndex:
    return TickerIndex.from_sec_json(client.company_tickers())


def fetch_filings(client: EdgarSource, cik: str, window_start: date) -> tuple[dict[str, Any], list[Filing]]:
    submissions = client.submissions(cik)
    blocks = [parse_filing_block(submissions["filings"]["recent"], filer_cik=cik)]
    for page in pages_overlapping(submissions, window_start):
        blocks.append(parse_filing_block(client.submissions_page(page), filer_cik=cik))
    return submissions, merge_filings(blocks)


def company_filings(
    client: EdgarSource, cik: str, window_start: date, predecessor_ciks: Sequence[str] = ()
) -> tuple[dict[str, Any], list[Filing], list[str]]:
    """The current registrant's submissions JSON plus filings merged across predecessors."""
    warnings: list[str] = []
    submissions, filings = fetch_filings(client, cik, window_start)
    if predecessor_ciks:
        lists = [filings]  # current registrant first, so its listing wins on shared accessions
        for old_cik in predecessor_ciks:
            try:
                lists.append(fetch_filings(client, old_cik, window_start)[1])
            except SecNotFoundError:
                warnings.append(f"predecessor CIK {old_cik} has no submissions on EDGAR")
        filings = merge_filings(lists)
    return submissions, filings, warnings


def snapshot_company(
    client: EdgarSource,
    index: TickerIndex,
    ticker: str,
    *,
    sector: str = "unknown",
    as_of: date,
    years: int = 3,
    predecessor_ciks: Sequence[str] = (),
) -> CompanySnapshot:
    """Cadence + financials for one ticker.

    `predecessor_ciks` are earlier registrants of the same company (e.g. after a holding-
    company reorganization, where SEC's ticker file only maps the new CIK). Their filings are
    merged into cadence; financials come from the current CIK's companyfacts, which carries
    prior-period comparatives.
    """
    entry = index.resolve(ticker)
    window_start = years_before(as_of, years)
    submissions, filings, warnings = company_filings(client, entry.cik, window_start, predecessor_ciks)
    cadence = compute_cadence(
        cik=entry.cik,
        name=str(submissions.get("name") or entry.title),
        fiscal_year_end=submissions.get("fiscalYearEnd"),
        filings=filings,
        as_of=as_of,
        years=years,
    )
    financials: FinancialsReport | None
    try:
        facts = client.company_facts(entry.cik)
        financials = compute_financials(facts, cik=entry.cik, name=cadence.name)
        warnings += financials.revenue.warnings + financials.net_income.warnings
    except SecNotFoundError:
        financials = None
        warnings.append("no XBRL company facts on EDGAR for this company")
    in_window = [f for f in filings if window_start <= f.filing_date <= as_of]
    return CompanySnapshot(
        entry=entry,
        sector=sector,
        submissions=submissions,
        filings=in_window,
        cadence=cadence,
        financials=financials,
        warnings=warnings,
    )


def persist_snapshot(session: Session, snap: CompanySnapshot) -> dict[str, int]:
    """Write one company's snapshot. Imported lazily so analysis works without a DB driver."""
    from filinglens.db import repository as repo

    repo.upsert_company(
        session,
        cik=snap.entry.cik,
        ticker=snap.entry.ticker,
        name=snap.cadence.name,
        sector=snap.sector,
        sic=snap.submissions.get("sic") or None,
        fiscal_year_end=snap.cadence.fiscal_year_end,
    )
    counts = {
        "filings": repo.upsert_filings(session, snap.entry.cik, snap.filings),
        "cadence_rows": repo.upsert_cadence(session, snap.cadence),
        "financial_rows": 0,
    }
    if snap.financials is not None:
        counts["financial_rows"] = repo.upsert_metric(
            session, snap.entry.cik, snap.financials.revenue
        ) + repo.upsert_metric(session, snap.entry.cik, snap.financials.net_income)
    return counts
