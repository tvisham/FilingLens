"""Risk-factor pipeline for one company: 10-Ks -> Item 1A -> chunks -> embeddings ->
storage -> classification -> year-over-year diff.

Classification is the only step that calls an LLM; it is optional (`classifier=None`) and
is logged as an `agent_runs` row through `log_run`. Re-running is idempotent: a section
already stored and fully classified for the same accession is reused unless `force`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from filinglens.agent.records import AgentRunResult
from filinglens.cadence import years_before
from filinglens.edgar.client import SecClient, SecNotFoundError
from filinglens.edgar.tickers import TickerIndex
from filinglens.pipeline import company_filings
from filinglens.risk.classify import ClassificationRun, Classifier, FactorInput, estimate_cost
from filinglens.risk.diff import FactorVector, diff_factors
from filinglens.risk.documents import extract_risk_section, recent_annual_reports
from filinglens.risk.embed import Embedder, mean_vector
from filinglens.risk.store import (
    CompanyRef,
    NewChunk,
    NewFactor,
    NewSection,
    RiskStore,
    StoredFactor,
    StoredSection,
)

CLASSIFY_SCENARIO = "classify_risk_factors"
MAX_WORDS_TO_CLASSIFY = 1200  # long factors are classified from their opening; citations stay valid


@dataclass
class SectionSummary:
    fiscal_year: int
    accession: str
    method: str
    split_method: str
    confidence: float
    needs_review: bool
    factors: int
    chunks: int
    reused: bool


@dataclass
class RiskIngestResult:
    ticker: str
    cik: str
    sections: list[SectionSummary] = field(default_factory=list)
    classification: ClassificationRun | None = None
    diff_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    estimated_cost_usd: Decimal | None = None


def _factor_inputs(factors: Sequence[StoredFactor]) -> list[FactorInput]:
    out = []
    for f in factors:
        words = f.text.split()
        text = " ".join(words[:MAX_WORDS_TO_CLASSIFY]) + (
            " [...]" if len(words) > MAX_WORDS_TO_CLASSIFY else ""
        )
        out.append(FactorInput(key=f"F{f.id}", heading=f.heading, text=text, chunk_ids=tuple(f.chunk_ids)))
    return out


def _classify(
    store: RiskStore,
    classifier: Classifier,
    ticker: str,
    factors: Sequence[StoredFactor],
    log_run: Callable[[AgentRunResult], int | None] | None,
) -> ClassificationRun:
    run = classifier.classify(_factor_inputs(factors))
    record = AgentRunResult(
        scenario=CLASSIFY_SCENARIO,
        input={"ticker": ticker, "factor_ids": [f.id for f in factors]},
        status=run.status,
        output={
            "classified": len(run.results),
            "failed": len(run.failures),
            "served_by": sorted(run.served_by),
        },
        error="; ".join(f"{k}: {v}" for k, v in list(run.failures.items())[:5]) or None,
        model=run.model,
        tokens_in=run.tokens_in,
        tokens_out=run.tokens_out,
        cost_usd=run.cost_usd,
        latency_ms=run.latency_ms,
        num_turns=run.requests,
    )
    run_id = log_run(record) if log_run else None
    to_id = {f"F{f.id}": f.id for f in factors}
    store.save_classifications(
        {to_id[k]: c for k, c in run.results.items()},
        {to_id[k]: reason for k, reason in run.failures.items()},
        model=", ".join(sorted(run.served_by)) or run.model,
        agent_run_id=run_id,
    )
    return run


def _vectors(store: RiskStore, section: StoredSection) -> list[FactorVector]:
    chunks = store.chunks(section.id)
    out = []
    for f in store.factors(section.id):
        vecs = [c.embedding for c in chunks if c.id in set(f.chunk_ids)]
        if vecs:
            out.append(FactorVector(f.id, f.heading, f.text, mean_vector(vecs)))
    return out


def ingest_company_risks(
    client: SecClient,
    store: RiskStore,
    embedder: Embedder,
    index: TickerIndex,
    ticker: str,
    *,
    sector: str = "unknown",
    predecessor_ciks: Sequence[str] = (),
    classifier: Classifier | None = None,
    log_run: Callable[[AgentRunResult], int | None] | None = None,
    as_of: date | None = None,
    years: int = 2,
    force: bool = False,
    max_cost_usd: Decimal | None = None,
) -> RiskIngestResult:
    entry = index.resolve(ticker)
    as_of = as_of or date.today()
    submissions, filings, warnings = company_filings(
        client, entry.cik, years_before(as_of, years + 1), predecessor_ciks
    )
    result = RiskIngestResult(ticker=entry.ticker, cik=entry.cik, warnings=list(warnings))
    try:
        facts = client.company_facts(entry.cik)
    except SecNotFoundError:
        facts = None
    company = CompanyRef(
        cik=entry.cik,
        ticker=entry.ticker,
        name=str(submissions.get("name") or entry.title),
        sector=sector,
        sic=submissions.get("sic") or None,
        fiscal_year_end=submissions.get("fiscalYearEnd") or None,
    )
    reports = recent_annual_reports(filings, entry.cik, facts, count=years)
    if not reports:
        result.warnings.append("no 10-K filings found")
        return result

    existing = {s.accession: s for s in store.sections(entry.cik)}
    stored_sections: list[StoredSection] = []
    to_classify: list[StoredFactor] = []
    for report in reports:
        prior = existing.get(report.accession)
        prior_factors = store.factors(prior.id) if prior else []
        done = (
            prior is not None
            and prior_factors
            and (classifier is None or all(f.status == "classified" for f in prior_factors))
        )
        if done and not force:
            assert prior is not None
            stored_sections.append(prior)
            result.sections.append(
                SectionSummary(
                    prior.fiscal_year, prior.accession, prior.method, prior.split_method, prior.confidence,
                    prior.needs_review, len(prior_factors), len(store.chunks(prior.id)), reused=True,
                )
            )  # fmt: skip
            continue

        extracted = extract_risk_section(client, report)
        vectors = embedder.embed_passages([c.embedding_text for c in extracted.chunks])
        section, factors = store.save_section(
            NewSection(
                company=company,
                accession=report.accession,
                filer_cik=report.filer_cik,
                primary_document=report.primary_document,
                form="10-K",
                filing_date=report.filing_date,
                report_date=report.report_date,
                fiscal_year=report.fiscal_year,
                source_url=extracted.source_url,
                method=extracted.method,
                split_method=extracted.split_method,
                confidence=extracted.confidence,
                needs_review=extracted.needs_review,
                word_count=extracted.word_count,
                text=extracted.text,
                embedding_model=embedder.name,
            ),
            [NewFactor(f.ordinal, f.heading, f.section_heading, f.text) for f in extracted.factors],
            [
                NewChunk(c.factor_ordinal, c.ordinal, c.heading, c.section_heading, c.text, c.token_count, v)
                for c, v in zip(extracted.chunks, vectors, strict=True)
            ],
        )
        stored_sections.append(section)
        to_classify += factors
        result.sections.append(
            SectionSummary(
                section.fiscal_year, section.accession, section.method, section.split_method,
                section.confidence, section.needs_review, len(factors), len(extracted.chunks), reused=False,
            )
        )  # fmt: skip
        if extracted.needs_review:
            result.warnings.append(
                f"FY{section.fiscal_year} Item 1A extraction confidence {section.confidence}: needs review"
            )

    if classifier is not None and to_classify:
        _, _, estimate = estimate_cost(classifier, _factor_inputs(to_classify))
        result.estimated_cost_usd = estimate
        remaining = None if max_cost_usd is None else max_cost_usd - classifier.spent_usd
        if remaining is not None and estimate > remaining:
            result.warnings.append(
                f"classification skipped: estimated ${estimate} exceeds the remaining budget ${remaining:.4f}"
            )
        else:
            result.classification = _classify(store, classifier, entry.ticker, to_classify, log_run)

    if len(stored_sections) >= 2:
        latest, prior_section = sorted(stored_sections, key=lambda s: -s.fiscal_year)[:2]
        rows = diff_factors(_vectors(store, prior_section), _vectors(store, latest))
        store.save_diffs(entry.cik, prior_section.fiscal_year, latest.fiscal_year, rows)
        for r in rows:
            result.diff_counts[r.status] = result.diff_counts.get(r.status, 0) + 1
    return result
