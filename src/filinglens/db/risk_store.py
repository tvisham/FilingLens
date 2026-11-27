"""`RiskStore` on Postgres + pgvector (cosine distance, HNSW index)."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Engine, delete, func, select, update
from sqlalchemy.orm import Session

from filinglens.cadence import Filing as FilingRow
from filinglens.db import repository as repo
from filinglens.db.models import Chunk, Company, RiskDiff, RiskFactorRow, RiskSection
from filinglens.db.session import session_scope
from filinglens.risk.classify import Classification
from filinglens.risk.diff import DiffRow
from filinglens.risk.store import (
    ChunkHit,
    NewChunk,
    NewFactor,
    NewSection,
    StoredChunk,
    StoredDiff,
    StoredFactor,
    StoredSection,
)


def _section(row: RiskSection, ticker: str) -> StoredSection:
    return StoredSection(
        id=row.id,
        cik=row.cik,
        ticker=ticker,
        accession=row.accession,
        fiscal_year=row.fiscal_year,
        report_date=row.report_date,
        source_url=row.source_url,
        method=row.method,
        split_method=row.split_method,
        confidence=row.extraction_confidence,
        needs_review=row.needs_review,
        word_count=row.word_count,
    )


def _chunk(row: Chunk) -> StoredChunk:
    return StoredChunk(
        id=row.id,
        section_id=row.risk_section_id,
        factor_ordinal=row.factor_ordinal,
        ordinal=row.ordinal,
        heading=row.heading,
        section_heading=row.section_heading,
        text=row.text,
        token_count=row.token_count,
        embedding=[float(x) for x in row.embedding],
    )


def _factor(row: RiskFactorRow) -> StoredFactor:
    return StoredFactor(
        id=row.id,
        section_id=row.risk_section_id,
        cik=row.cik,
        fiscal_year=row.fiscal_year,
        ordinal=row.ordinal,
        heading=row.heading,
        section_heading=row.section_heading,
        text=row.text,
        chunk_ids=list(row.chunk_ids),
        status=row.status,
        cited_chunk_ids=list(row.cited_chunk_ids),
        categories=list(row.categories),
        severity=row.severity,
        summary=row.summary,
        model=row.model,
        agent_run_id=row.agent_run_id,
        failure=row.failure,
    )


class SqlRiskStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @staticmethod
    def _tickers(session: Session, ciks: Sequence[str]) -> dict[str, str]:
        rows = session.execute(select(Company.cik, Company.ticker).where(Company.cik.in_(list(ciks))))
        return {cik: ticker for cik, ticker in rows.tuples()}

    def save_section(
        self, section: NewSection, factors: Sequence[NewFactor], chunks: Sequence[NewChunk]
    ) -> tuple[StoredSection, list[StoredFactor]]:
        c = section.company
        with session_scope(self._engine) as session:
            repo.upsert_company(
                session,
                cik=c.cik,
                ticker=c.ticker,
                name=c.name,
                sector=c.sector,
                sic=c.sic,
                fiscal_year_end=c.fiscal_year_end,
            )
            repo.upsert_filings(
                session,
                c.cik,
                [
                    FilingRow(
                        accession=section.accession,
                        form=section.form,
                        filing_date=section.filing_date,
                        report_date=section.report_date,
                        primary_document=section.primary_document,
                        filer_cik=section.filer_cik,
                    )
                ],
            )
            session.execute(delete(RiskSection).where(RiskSection.accession == section.accession))
            row = RiskSection(
                cik=c.cik,
                accession=section.accession,
                fiscal_year=section.fiscal_year,
                report_date=section.report_date,
                source_url=section.source_url,
                method=section.method,
                split_method=section.split_method,
                extraction_confidence=section.confidence,
                needs_review=section.needs_review,
                word_count=section.word_count,
                text=section.text,
                embedding_model=section.embedding_model,
            )
            session.add(row)
            session.flush()
            chunk_rows = [
                Chunk(
                    risk_section_id=row.id,
                    factor_ordinal=ch.factor_ordinal,
                    ordinal=ch.ordinal,
                    heading=ch.heading,
                    section_heading=ch.section_heading,
                    text=ch.text,
                    token_count=ch.token_count,
                    embedding=list(ch.embedding),
                )
                for ch in chunks
            ]
            session.add_all(chunk_rows)
            session.flush()
            ids: dict[int, list[int]] = {}
            for ch in chunk_rows:
                ids.setdefault(ch.factor_ordinal, []).append(ch.id)
            factor_rows = [
                RiskFactorRow(
                    risk_section_id=row.id,
                    cik=c.cik,
                    fiscal_year=section.fiscal_year,
                    ordinal=f.ordinal,
                    heading=f.heading,
                    section_heading=f.section_heading,
                    text=f.text,
                    chunk_ids=ids.get(f.ordinal, []),
                    status="unclassified",
                    categories=[],
                    cited_chunk_ids=[],
                )
                for f in factors
            ]
            session.add_all(factor_rows)
            session.flush()
            return _section(row, c.ticker), [_factor(f) for f in factor_rows]

    def save_classifications(
        self,
        results: dict[int, Classification],
        failures: dict[int, str],
        *,
        model: str,
        agent_run_id: int | None,
    ) -> None:
        with session_scope(self._engine) as session:
            for fid, cl in results.items():
                session.execute(
                    update(RiskFactorRow)
                    .where(RiskFactorRow.id == fid)
                    .values(
                        status="classified",
                        categories=[str(x) for x in cl.categories],
                        severity=cl.severity,
                        summary=cl.summary,
                        cited_chunk_ids=list(cl.chunk_ids),
                        model=model,
                        agent_run_id=agent_run_id,
                        failure=None,
                    )
                )
            for fid, reason in failures.items():
                session.execute(
                    update(RiskFactorRow)
                    .where(RiskFactorRow.id == fid)
                    .values(status="failed", model=model, agent_run_id=agent_run_id, failure=reason)
                )

    def sections(self, cik: str) -> list[StoredSection]:
        with session_scope(self._engine) as session:
            rows = session.scalars(
                select(RiskSection).where(RiskSection.cik == cik).order_by(RiskSection.fiscal_year.desc())
            ).all()
            tickers = self._tickers(session, [cik])
            return [_section(r, tickers.get(r.cik, "")) for r in rows]

    def section_by_id(self, section_id: int) -> StoredSection | None:
        with session_scope(self._engine) as session:
            row = session.get(RiskSection, section_id)
            if row is None:
                return None
            return _section(row, self._tickers(session, [row.cik]).get(row.cik, ""))

    def factors(self, section_id: int) -> list[StoredFactor]:
        with session_scope(self._engine) as session:
            rows = session.scalars(
                select(RiskFactorRow)
                .where(RiskFactorRow.risk_section_id == section_id)
                .order_by(RiskFactorRow.ordinal)
            ).all()
            return [_factor(r) for r in rows]

    def chunks(self, section_id: int) -> list[StoredChunk]:
        with session_scope(self._engine) as session:
            rows = session.scalars(
                select(Chunk).where(Chunk.risk_section_id == section_id).order_by(Chunk.ordinal)
            ).all()
            return [_chunk(r) for r in rows]

    def chunks_by_id(self, ids: Sequence[int]) -> list[StoredChunk]:
        if not ids:
            return []
        with session_scope(self._engine) as session:
            rows = {r.id: r for r in session.scalars(select(Chunk).where(Chunk.id.in_(list(ids)))).all()}
            return [_chunk(rows[i]) for i in ids if i in rows]

    def save_diffs(self, cik: str, from_year: int, to_year: int, rows: Sequence[DiffRow]) -> None:
        with session_scope(self._engine) as session:
            session.execute(delete(RiskDiff).where(RiskDiff.cik == cik))
            session.add_all(
                RiskDiff(
                    cik=cik,
                    from_year=from_year,
                    to_year=to_year,
                    status=r.status,
                    from_factor_id=r.from_id,
                    to_factor_id=r.to_id,
                    similarity=r.similarity,
                    text_similarity=r.text_similarity,
                )
                for r in rows
            )

    def diffs(self, cik: str) -> list[StoredDiff]:
        with session_scope(self._engine) as session:
            rows = session.scalars(select(RiskDiff).where(RiskDiff.cik == cik).order_by(RiskDiff.id)).all()
            return [
                StoredDiff(
                    r.cik,
                    r.from_year,
                    r.to_year,
                    r.status,
                    r.from_factor_id,
                    r.to_factor_id,
                    r.similarity,
                    r.text_similarity,
                )
                for r in rows
            ]

    def search_chunks(
        self, query: Sequence[float], *, ciks: Sequence[str] | None = None, limit: int = 10
    ) -> list[ChunkHit]:
        with session_scope(self._engine) as session:
            latest = select(RiskSection.cik, func.max(RiskSection.fiscal_year).label("fy")).group_by(
                RiskSection.cik
            )
            if ciks is not None:
                latest = latest.where(RiskSection.cik.in_(list(ciks)))
            latest_sq = latest.subquery()
            distance = Chunk.embedding.cosine_distance(list(query)).label("distance")
            stmt = (
                select(Chunk, RiskSection, distance)
                .join(RiskSection, Chunk.risk_section_id == RiskSection.id)
                .join(
                    latest_sq,
                    (latest_sq.c.cik == RiskSection.cik) & (latest_sq.c.fy == RiskSection.fiscal_year),
                )
                .order_by(distance)
                .limit(limit)
            )
            rows = session.execute(stmt).all()
            tickers = self._tickers(session, list({s.cik for _, s, _ in rows}))
            return [
                ChunkHit(_chunk(ch), _section(sec, tickers.get(sec.cik, "")), 1.0 - float(dist))
                for ch, sec, dist in rows
            ]
