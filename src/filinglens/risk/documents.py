"""Find a company's recent 10-Ks and extract each one's Item 1A.

Only original 10-Ks count (not 10-K/A). The fiscal year is taken from the XBRL `fy` of the
10-K's own facts, because companies label years inconsistently (Walmart calls the year
ending 2026-01-31 "fiscal 2026"; Home Depot calls the year ending 2026-02-01 "fiscal
2025"). Without XBRL, the period-end year is used.

Some filers incorporate risk factors by reference from the annual report filed as
Exhibit 13 (e.g. Wells Fargo). When the primary document's Item 1A is missing or only a
cross-reference, the filing folder is listed (EDGAR's documented `index.json`) and an
`ex13`-style document is searched for a "Risk Factors" section.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from filinglens.cadence import Filing, filing_url
from filinglens.edgar.client import SecClient, SecNotFoundError
from filinglens.risk.chunk import Chunk, RiskFactor, chunk_factors, split_factors
from filinglens.risk.extract import MIN_WORDS, Section, confidence, locate_item_1a
from filinglens.risk.html import html_to_blocks

REVIEW_THRESHOLD = 0.7
_EX13 = re.compile(r"ex[-_]?13(?!\d)|exhibit[-_]?13(?!\d)", re.I)
_EXHIBIT = re.compile(r"(^|[-_x])ex(hibit)?[-_]?\d", re.I)
_XBRL_VIEWER = re.compile(r"^R\d+\.htm$", re.I)


@dataclass(frozen=True)
class AnnualReport:
    accession: str
    filer_cik: str
    report_date: date
    filing_date: date
    fiscal_year: int
    primary_document: str


@dataclass(frozen=True)
class RiskSectionResult:
    report: AnnualReport
    source_url: str  # the document the section was taken from
    method: str  # item_1a | risk_factors_heading | not_found
    split_method: str  # headings | windows
    confidence: float
    word_count: int
    text: str
    factors: list[RiskFactor]
    chunks: list[Chunk]

    @property
    def needs_review(self) -> bool:
        return self.confidence < REVIEW_THRESHOLD


def fiscal_year_for(accession: str, report_date: date, companyfacts: Mapping[str, Any] | None) -> int:
    for taxonomy in (companyfacts or {}).get("facts", {}).values():
        for concept in taxonomy.values():
            for rows in concept.get("units", {}).values():
                for row in rows:
                    if row.get("accn") == accession and row.get("form") == "10-K" and row.get("fy"):
                        return int(row["fy"])
    return report_date.year


def recent_annual_reports(
    filings: Sequence[Filing], company_cik: str, companyfacts: Mapping[str, Any] | None, count: int = 2
) -> list[AnnualReport]:
    """The `count` most recent original 10-Ks, newest first, one per fiscal period."""
    seen: set[date] = set()
    out: list[AnnualReport] = []
    for f in sorted(filings, key=lambda f: (f.report_date or f.filing_date, f.filing_date), reverse=True):
        if f.form != "10-K" or f.report_date is None or not f.primary_document or f.report_date in seen:
            continue
        seen.add(f.report_date)
        out.append(
            AnnualReport(
                accession=f.accession,
                filer_cik=f.filer_cik or company_cik,
                report_date=f.report_date,
                filing_date=f.filing_date,
                fiscal_year=fiscal_year_for(f.accession, f.report_date, companyfacts),
                primary_document=f.primary_document,
            )
        )
        if len(out) == count:
            break
    return out


def _alternative_document_urls(client: SecClient, report: AnnualReport, limit: int = 3) -> list[str]:
    """Other documents in the filing that may hold the annual report: an Exhibit 13 first,
    then non-exhibit HTML documents by size (Wells Fargo lists a short cross-reference
    wrapper as the primary document and files the full annual report next to it)."""
    try:
        listing = client.filing_index(report.filer_cik, report.accession)
    except SecNotFoundError:
        return []
    ex13: list[str] = []
    others: list[tuple[int, str]] = []
    for item in listing.get("directory", {}).get("item", []):
        name = str(item.get("name", ""))
        lower = name.lower()
        if not lower.endswith((".htm", ".html")) or name == report.primary_document:
            continue
        if _EX13.search(name):
            ex13.append(name)
        elif not (_XBRL_VIEWER.match(name) or "index" in lower or _EXHIBIT.search(name)):
            size = int(item.get("size") or 0)
            others.append((size, name))
    names = ex13 + [n for _, n in sorted(others, reverse=True)]
    return [filing_url(report.filer_cik, report.accession, n) for n in names[:limit]]


def _analyse(section: Section) -> tuple[list[RiskFactor], str, float]:
    factors, split_method = split_factors(section.blocks)
    score = confidence(section, len(factors) if split_method == "headings" else 0)
    return factors, split_method, score


def extract_risk_section(client: SecClient, report: AnnualReport) -> RiskSectionResult:
    primary = filing_url(report.filer_cik, report.accession, report.primary_document)
    section = locate_item_1a(html_to_blocks(client.document(primary)))
    source = primary
    if section.word_count < MIN_WORDS:  # missing, or a cross-reference to the annual report
        for url in _alternative_document_urls(client, report):
            candidate = locate_item_1a(html_to_blocks(client.document(url)))
            if candidate.word_count > section.word_count:
                section, source = candidate, url
    factors, split_method, score = _analyse(section)
    if source != primary:
        score = min(score, 0.6)  # not the primary 10-K document: always worth a human look
    return RiskSectionResult(
        report=report,
        source_url=source,
        method=section.method,
        split_method=split_method,
        confidence=score,
        word_count=section.word_count,
        text=section.text,
        factors=factors,
        chunks=chunk_factors(factors),
    )
