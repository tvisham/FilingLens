"""The FilingLens tools as plain functions over the SEC client.

Kept separate from the MCP wiring (`server.py`) so they are unit-testable without a
protocol round trip. Every function is read-only: it reads SEC data through the
rate-limited, cached `SecClient` (or the risk store) and returns a Pydantic model.
Numbers come from the deterministic Phase 1 code; risk classifications come from the
stored Phase 3 pipeline output. No tool calls an LLM.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from filinglens.cadence import Filing, compute_cadence, filing_url, years_before
from filinglens.config import Universe
from filinglens.edgar.client import SecClient, SecNotFoundError
from filinglens.edgar.tickers import TickerEntry, TickerIndex
from filinglens.financials import MetricResult, compute_financials
from filinglens.mcp_server.schemas import (
    ChunkHitOut,
    ChunkOut,
    ChunksResult,
    CompanyInfo,
    CompanyRiskSummary,
    FactorRef,
    FilingIndex,
    FilingRef,
    FinancialsResult,
    FiscalYearCheck,
    FormCount,
    MetricSeries,
    PriceHistoryResult,
    PricePoint,
    QuarterValue,
    RiskChange,
    RiskChangesResult,
    RiskFactorOut,
    RiskFactorsResult,
    RiskSearchResult,
    SearchResult,
    SectorRiskResult,
)
from filinglens.pipeline import company_filings, load_ticker_index
from filinglens.prices.alphavantage import MAX_SHARE_COUNT_AGE_DAYS, PriceBar, latest_share_count, market_cap
from filinglens.prices.alphavantage import SOURCE as PRICE_SOURCE
from filinglens.risk.embed import Embedder
from filinglens.risk.store import RiskStore, StoredFactor, StoredSection

MAX_YEARS = 10
MAX_LIMIT = 100
MAX_QUARTERS = 8
MAX_CHUNKS = 10
EXCERPT_CHARS = 600
NOT_INGESTED = (
    "Risk factors for {ticker} have not been ingested (run `filinglens risk ingest {ticker}`). "
    "Do not describe this company's risk factors; say they are not available."
)


class ToolInputError(ValueError):
    """A request the caller can fix (unknown ticker, no data for it). Shown to the model."""


@dataclass
class ToolContext:
    client: SecClient
    universe: Universe | None = None
    today: Callable[[], date] = date.today
    store: RiskStore | None = None
    embedder_factory: Callable[[], Embedder] | None = None
    price_loader: Callable[[str, int], list[PriceBar]] | None = None  # (cik, days) -> bars
    _index: TickerIndex | None = field(default=None, init=False, repr=False)
    _embedder: Embedder | None = field(default=None, init=False, repr=False)

    def embedder(self) -> Embedder:
        if self._embedder is None:
            if self.embedder_factory is None:
                raise ToolInputError("semantic search is not configured on this server")
            self._embedder = self.embedder_factory()
        return self._embedder

    def risk_store(self) -> RiskStore:
        if self.store is None:
            raise ToolInputError("the risk-factor database is not configured (DATABASE_URL)")
        return self.store

    def resolve(self, ticker: str) -> TickerEntry:
        if self._index is None:
            self._index = load_ticker_index(self.client)
        try:
            return self._index.resolve(ticker)
        except KeyError as exc:  # UnknownTickerError
            raise ToolInputError(
                f"unknown ticker {ticker!r}: not in SEC's company_tickers.json "
                "(the company may not file with the SEC)"
            ) from exc

    def predecessors(self, ticker: str) -> list[str]:
        return self.universe.predecessors_for(ticker) if self.universe else []

    def sector(self, ticker: str) -> str | None:
        if self.universe is None:
            return None
        want = ticker.strip().upper()
        return next((c.sector for c in self.universe.companies() if c.ticker == want), None)


def _ref(f: Filing, company_cik: str) -> FilingRef:
    return FilingRef(
        accession=f.accession,
        form=f.form,
        filing_date=f.filing_date,
        report_date=f.report_date,
        items=f.items,
        description=f.description,
        url=filing_url(f.filer_cik or company_cik, f.accession, f.primary_document),
    )


def _wants(f: Filing, forms: Sequence[str] | None) -> bool:
    if not forms:
        return True
    wanted = {x.strip().upper() for x in forms}
    return f.form.upper() in wanted or f.base_form.upper() in wanted


# -- tools -------------------------------------------------------------------------------


def lookup_company(ctx: ToolContext, ticker: str) -> CompanyInfo:
    entry = ctx.resolve(ticker)
    subs = ctx.client.submissions(entry.cik)
    sector = ctx.sector(entry.ticker)
    return CompanyInfo(
        ticker=entry.ticker,
        cik=entry.cik,
        name=str(subs.get("name") or entry.title),
        in_universe=sector is not None,
        sector=sector,
        sic=subs.get("sic") or None,
        sic_description=subs.get("sicDescription") or None,
        fiscal_year_end=subs.get("fiscalYearEnd") or None,
        exchanges=[e for e in subs.get("exchanges") or [] if e],
        former_names=[str(n.get("name")) for n in subs.get("formerNames") or [] if n.get("name")],
        predecessor_ciks=ctx.predecessors(entry.ticker),
    )


def get_filing_index(
    ctx: ToolContext,
    ticker: str,
    forms: Sequence[str] | None = None,
    years: int = 3,
    limit: int = 20,
) -> FilingIndex:
    entry = ctx.resolve(ticker)
    years = max(1, min(years, MAX_YEARS))
    limit = max(1, min(limit, MAX_LIMIT))
    as_of = ctx.today()
    window_start = years_before(as_of, years)
    subs, filings, warnings = company_filings(
        ctx.client, entry.cik, window_start, ctx.predecessors(entry.ticker)
    )
    name = str(subs.get("name") or entry.title)
    cadence = compute_cadence(
        cik=entry.cik,
        name=name,
        fiscal_year_end=subs.get("fiscalYearEnd"),
        filings=filings,
        as_of=as_of,
        years=years,
    )
    matching = sorted(
        (f for f in filings if window_start <= f.filing_date <= as_of and _wants(f, forms)),
        key=lambda f: (f.filing_date, f.accession),
        reverse=True,
    )
    return FilingIndex(
        ticker=entry.ticker,
        cik=entry.cik,
        name=name,
        fiscal_year_end=cadence.fiscal_year_end,
        window_start=cadence.window_start,
        window_end=cadence.window_end,
        counts=[
            FormCount(
                form=s.form,
                count=s.count,
                amendments=s.amendments,
                latest_filing_date=max(s.dates) if s.dates else None,
                median_days_after_period_end=s.median_days_after_period_end,
            )
            for s in cadence.forms
        ],
        fiscal_years=[
            FiscalYearCheck(
                previous_annual_period_end=p.period_start_exclusive,
                annual_period_end=p.period_end,
                ten_q_count=p.ten_q_count,
                expected=p.expected,
                complete=p.complete,
                in_progress=p.in_progress,
            )
            for p in cadence.fiscal_periods
        ],
        filings=[_ref(f, entry.cik) for f in matching[:limit]],
        total_matching=len(matching),
        warnings=warnings,
    )


def _series(result: MetricResult, quarters: int) -> MetricSeries:
    return MetricSeries(
        metric=result.metric.value,
        concept=result.concept,
        quarters=[
            QuarterValue(
                fiscal_year=q.fiscal_year,
                fiscal_quarter=q.fiscal_quarter,
                period_start=q.period_start,
                period_end=q.period_end,
                value_usd=q.value,
                concept=q.concept,
                accession=q.accession,
                form=q.form,
                derived=q.derived,
                derivation=q.derivation,
                source_accessions=q.source_accessions,
            )
            for q in result.quarters[-quarters:]
        ],
        warnings=result.warnings,
    )


def get_financials(ctx: ToolContext, ticker: str, quarters: int = 2) -> FinancialsResult:
    entry = ctx.resolve(ticker)
    quarters = max(1, min(quarters, MAX_QUARTERS))
    try:
        facts = ctx.client.company_facts(entry.cik)
    except SecNotFoundError as exc:
        raise ToolInputError(f"{entry.ticker}: no XBRL financial data on EDGAR") from exc
    report = compute_financials(facts, cik=entry.cik, name=str(facts.get("entityName") or entry.title))
    return FinancialsResult(
        ticker=entry.ticker,
        cik=entry.cik,
        name=report.name,
        revenue=_series(report.revenue, quarters),
        net_income=_series(report.net_income, quarters),
    )


def _factor_ref(f: StoredFactor) -> FactorRef:
    return FactorRef(
        factor_id=f.id,
        heading=f.heading,
        categories=f.categories,
        severity=f.severity,
        summary=f.summary,
        cited_chunk_ids=f.cited_chunk_ids,
    )


def _section_for(store: RiskStore, cik: str, fiscal_year: int | None) -> StoredSection | None:
    sections = store.sections(cik)
    if fiscal_year is None:
        return sections[0] if sections else None
    return next((s for s in sections if s.fiscal_year == fiscal_year), None)


def get_risk_factors(ctx: ToolContext, ticker: str, fiscal_year: int | None = None) -> RiskFactorsResult:
    entry = ctx.resolve(ticker)
    store = ctx.risk_store()
    section = _section_for(store, entry.cik, fiscal_year)
    if section is None:
        years = sorted({s.fiscal_year for s in store.sections(entry.cik)}, reverse=True)
        message = NOT_INGESTED.format(ticker=entry.ticker)
        if years and fiscal_year is not None:
            message = f"No risk factors stored for FY{fiscal_year}; available fiscal years: {years}."
        return RiskFactorsResult(ticker=entry.ticker, status="not_available", message=message)
    factors = store.factors(section.id)
    classified = sum(f.status == "classified" for f in factors)
    message = f"{len(factors)} risk factors from the FY{section.fiscal_year} 10-K; {classified} classified."
    if section.needs_review:
        message += " The extraction has low confidence and needs human review: say so."
    return RiskFactorsResult(
        ticker=entry.ticker,
        status="available",
        message=message,
        fiscal_year=section.fiscal_year,
        accession=section.accession,
        source_url=section.source_url,
        extraction_confidence=section.confidence,
        needs_review=section.needs_review,
        factors=[
            RiskFactorOut(
                factor_id=f.id,
                heading=f.heading,
                section_heading=f.section_heading,
                status=f.status,
                categories=f.categories,
                severity=f.severity,
                summary=f.summary,
                cited_chunk_ids=f.cited_chunk_ids,
                chunk_ids=f.chunk_ids,
            )
            for f in factors
        ],
    )


def get_risk_changes(ctx: ToolContext, ticker: str, limit: int = 50) -> RiskChangesResult:
    entry = ctx.resolve(ticker)
    store = ctx.risk_store()
    rows = store.diffs(entry.cik)
    if not rows:
        return RiskChangesResult(
            ticker=entry.ticker,
            status="not_available",
            message=NOT_INGESTED.format(ticker=entry.ticker)
            + " A year-over-year comparison needs two ingested 10-Ks.",
        )
    by_id: dict[int, StoredFactor] = {}
    for section in store.sections(entry.cik):
        by_id |= {f.id: f for f in store.factors(section.id)}
    counts: dict[str, int] = {}
    changes: list[RiskChange] = []
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.status == "unchanged":
            continue
        prior = by_id.get(r.from_factor_id) if r.from_factor_id is not None else None
        latest = by_id.get(r.to_factor_id) if r.to_factor_id is not None else None
        changes.append(
            RiskChange(
                status=r.status,
                prior=_factor_ref(prior) if prior else None,
                latest=_factor_ref(latest) if latest else None,
                match_score=r.similarity,
                text_similarity=r.text_similarity,
            )
        )
    return RiskChangesResult(
        ticker=entry.ticker,
        status="available",
        message=f"Risk factors compared between FY{rows[0].from_year} and FY{rows[0].to_year} 10-Ks.",
        from_year=rows[0].from_year,
        to_year=rows[0].to_year,
        counts=counts,
        changes=changes[: max(1, min(limit, MAX_LIMIT))],
    )


def search_risk_factors(
    ctx: ToolContext,
    query: str,
    tickers: Sequence[str] | None = None,
    sector: str | None = None,
    limit: int = 10,
) -> RiskSearchResult:
    store = ctx.risk_store()
    if not query.strip():
        raise ToolInputError("query is empty")
    wanted = [t.strip().upper() for t in tickers or []]
    if sector:
        if ctx.universe is None:
            raise ToolInputError("no company universe configured, so sectors are unknown")
        in_sector = [c.ticker for c in ctx.universe.companies() if c.sector == sector.strip().lower()]
        if not in_sector:
            raise ToolInputError(f"unknown sector {sector!r}; known: {sorted(ctx.universe.sectors)}")
        wanted = [t for t in wanted if t in in_sector] if wanted else in_sector
    ciks = [ctx.resolve(t).cik for t in wanted] if wanted else None
    hits = store.search_chunks(ctx.embedder().embed_query(query), ciks=ciks, limit=max(1, min(limit, 25)))
    return RiskSearchResult(
        query=query,
        tickers=wanted,
        hits=[
            ChunkHitOut(
                chunk_id=h.chunk.id,
                ticker=h.section.ticker,
                fiscal_year=h.section.fiscal_year,
                heading=h.chunk.heading,
                excerpt=h.chunk.text[:EXCERPT_CHARS],
                score=round(h.score, 4),
            )
            for h in hits
        ],
    )


def wrap_filing_text(text: str) -> str:
    """Delimit quoted filing text so it cannot pose as instructions or close the wrapper."""
    return "<filing_text>" + text.replace("<", "&lt;").replace(">", "&gt;") + "</filing_text>"


def get_chunks(ctx: ToolContext, chunk_ids: Sequence[int]) -> ChunksResult:
    store = ctx.risk_store()
    ids = list(dict.fromkeys(chunk_ids))[:MAX_CHUNKS]
    sections: dict[int, StoredSection | None] = {}
    out: list[ChunkOut] = []
    for c in store.chunks_by_id(ids):
        if c.section_id not in sections:
            sections[c.section_id] = store.section_by_id(c.section_id)
        section = sections[c.section_id]
        if section is None:
            continue
        out.append(
            ChunkOut(
                chunk_id=c.id,
                ticker=section.ticker,
                fiscal_year=section.fiscal_year,
                heading=c.heading,
                text=wrap_filing_text(c.text),
                source_url=section.source_url,
            )
        )
    got = {c.chunk_id for c in out}
    return ChunksResult(chunks=out, missing_ids=[i for i in ids if i not in got])


def compare_sector_risks(ctx: ToolContext, sector: str, top: int = 3) -> SectorRiskResult:
    store = ctx.risk_store()
    if ctx.universe is None:
        raise ToolInputError("no company universe configured, so sectors are unknown")
    name = sector.strip().lower()
    tickers = ctx.universe.sectors.get(name)
    if not tickers:
        raise ToolInputError(f"unknown sector {sector!r}; known: {sorted(ctx.universe.sectors)}")
    companies: list[CompanyRiskSummary] = []
    totals: dict[str, int] = {}
    for ticker in tickers:
        entry = ctx.resolve(ticker)
        section = _section_for(store, entry.cik, None)
        if section is None:
            companies.append(
                CompanyRiskSummary(
                    ticker=entry.ticker,
                    fiscal_year=None,
                    status="not_available",
                    classified_factors=0,
                    category_counts={},
                    mean_severity=None,
                    top_risks=[],
                )
            )
            continue
        factors = [f for f in store.factors(section.id) if f.status == "classified"]
        counts: dict[str, int] = {}
        for f in factors:
            for cat in f.categories:
                counts[cat] = counts.get(cat, 0) + 1
                totals[cat] = totals.get(cat, 0) + 1
        severities = [f.severity for f in factors if f.severity is not None]
        ranked = sorted(factors, key=lambda f: (-(f.severity or 0), f.ordinal))
        companies.append(
            CompanyRiskSummary(
                ticker=entry.ticker,
                fiscal_year=section.fiscal_year,
                status="available",
                classified_factors=len(factors),
                category_counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                mean_severity=round(sum(severities) / len(severities), 2) if severities else None,
                top_risks=[_factor_ref(f) for f in ranked[: max(1, min(top, 10))]],
                needs_review=section.needs_review,
            )
        )
    return SectorRiskResult(
        sector=name,
        companies=companies,
        category_totals=dict(sorted(totals.items(), key=lambda kv: -kv[1])),
    )


def search_filings(
    ctx: ToolContext,
    ticker: str,
    query: str = "",
    forms: Sequence[str] | None = None,
    items: Sequence[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 25,
) -> SearchResult:
    entry = ctx.resolve(ticker)
    limit = max(1, min(limit, MAX_LIMIT))
    end = end_date or ctx.today()
    start = start_date or years_before(end, 3)
    if start > end:
        raise ToolInputError(f"start_date {start} is after end_date {end}")
    start = max(start, years_before(end, MAX_YEARS))
    _, filings, _ = company_filings(ctx.client, entry.cik, start, ctx.predecessors(entry.ticker))
    terms = [t for t in query.lower().split() if t]
    wanted_items = {i.strip() for i in items or [] if i.strip()}

    def matches(f: Filing) -> bool:
        if not (start <= f.filing_date <= end and _wants(f, forms)):
            return False
        if wanted_items and not wanted_items.intersection(f.items):
            return False
        haystack = " ".join(
            [f.form, f.description or "", " ".join(f.items), f.primary_document or ""]
        ).lower()
        return all(t in haystack for t in terms)

    hits = sorted(filter(matches, filings), key=lambda f: (f.filing_date, f.accession), reverse=True)
    return SearchResult(
        ticker=entry.ticker,
        cik=entry.cik,
        query=query,
        search_scope=(
            f"filing-index metadata (form, 8-K items, document description and name) for "
            f"{start} to {end}; not the documents' full text"
        ),
        total_matching=len(hits),
        filings=[_ref(f, entry.cik) for f in hits[:limit]],
    )


def get_price_history(ctx: ToolContext, ticker: str, days: int = 30) -> PriceHistoryResult:
    entry = ctx.resolve(ticker)
    if ctx.price_loader is None:
        return PriceHistoryResult(
            ticker=entry.ticker, status="not_available", message="price data is not configured on this server"
        )
    bars = ctx.price_loader(entry.cik, max(1, min(days, 100)))
    if not bars:
        return PriceHistoryResult(
            ticker=entry.ticker,
            status="not_available",
            message=f"no prices stored for {entry.ticker} (run `filinglens prices fetch {entry.ticker}`)",
        )
    last = bars[-1]
    try:
        shares = latest_share_count(
            ctx.client.company_facts(entry.cik),
            not_before=last.day - timedelta(days=MAX_SHARE_COUNT_AGE_DAYS),
        )
    except SecNotFoundError:
        shares = None
    return PriceHistoryResult(
        ticker=entry.ticker,
        status="available",
        message=f"{len(bars)} trading days to {last.day}."
        + ("" if shares else " No current share count in SEC XBRL, so no market cap."),
        source=PRICE_SOURCE,
        last_trading_day=last.day,
        last_close_usd=last.close,
        closes=[PricePoint(day=b.day, close=b.close, volume=b.volume) for b in bars],
        shares_outstanding=shares.shares if shares else None,
        shares_as_of=shares.as_of if shares else None,
        shares_source=f"{shares.concept} ({shares.accession})" if shares else None,
        market_cap_usd=market_cap(last.close, shares) if shares else None,
    )
