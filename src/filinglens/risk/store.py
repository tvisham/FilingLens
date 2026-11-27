"""Storage interface for risk sections, chunks, factors and diffs.

`SqlRiskStore` (filinglens.db.risk_store) is the real one: Postgres + pgvector. The
in-memory store here backs tests and `--no-db` runs; it searches vectors by brute force.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Protocol

from filinglens.risk.classify import Classification
from filinglens.risk.diff import DiffRow
from filinglens.risk.embed import cosine


@dataclass(frozen=True)
class CompanyRef:
    cik: str
    ticker: str
    name: str
    sector: str
    sic: str | None = None
    fiscal_year_end: str | None = None


@dataclass(frozen=True)
class NewChunk:
    factor_ordinal: int
    ordinal: int
    heading: str | None
    section_heading: str | None
    text: str
    token_count: int
    embedding: Sequence[float]


@dataclass(frozen=True)
class NewFactor:
    ordinal: int
    heading: str | None
    section_heading: str | None
    text: str


@dataclass(frozen=True)
class NewSection:
    company: CompanyRef
    accession: str
    filer_cik: str
    primary_document: str
    form: str
    filing_date: date
    report_date: date
    fiscal_year: int
    source_url: str
    method: str
    split_method: str
    confidence: float
    needs_review: bool
    word_count: int
    text: str
    embedding_model: str


@dataclass(frozen=True)
class StoredSection:
    id: int
    cik: str
    ticker: str
    accession: str
    fiscal_year: int
    report_date: date
    source_url: str
    method: str
    split_method: str
    confidence: float
    needs_review: bool
    word_count: int


@dataclass(frozen=True)
class StoredChunk:
    id: int
    section_id: int
    factor_ordinal: int
    ordinal: int
    heading: str | None
    section_heading: str | None
    text: str
    token_count: int
    embedding: Sequence[float] = field(repr=False)


@dataclass(frozen=True)
class StoredFactor:
    id: int
    section_id: int
    cik: str
    fiscal_year: int
    ordinal: int
    heading: str | None
    section_heading: str | None
    text: str
    chunk_ids: list[int]  # every chunk of this factor
    status: str = "unclassified"  # unclassified | classified | failed
    cited_chunk_ids: list[int] = field(default_factory=list)  # the chunks the summary cites
    categories: list[str] = field(default_factory=list)
    severity: int | None = None
    summary: str | None = None
    model: str | None = None
    agent_run_id: int | None = None
    failure: str | None = None


@dataclass(frozen=True)
class StoredDiff:
    cik: str
    from_year: int
    to_year: int
    status: str
    from_factor_id: int | None
    to_factor_id: int | None
    similarity: float | None
    text_similarity: float | None


@dataclass(frozen=True)
class ChunkHit:
    chunk: StoredChunk
    section: StoredSection
    score: float


class RiskStore(Protocol):
    def save_section(
        self, section: NewSection, factors: Sequence[NewFactor], chunks: Sequence[NewChunk]
    ) -> tuple[StoredSection, list[StoredFactor]]:
        """Replace any stored section for the same accession; factors get their chunk ids."""
        ...

    def save_classifications(
        self,
        results: dict[int, Classification],
        failures: dict[int, str],
        *,
        model: str,
        agent_run_id: int | None,
    ) -> None: ...

    def sections(self, cik: str) -> list[StoredSection]:
        """Newest fiscal year first."""
        ...

    def section_by_id(self, section_id: int) -> StoredSection | None: ...

    def factors(self, section_id: int) -> list[StoredFactor]: ...

    def chunks(self, section_id: int) -> list[StoredChunk]: ...

    def chunks_by_id(self, ids: Sequence[int]) -> list[StoredChunk]: ...

    def save_diffs(self, cik: str, from_year: int, to_year: int, rows: Sequence[DiffRow]) -> None: ...

    def diffs(self, cik: str) -> list[StoredDiff]:
        """The diff between the two most recent fiscal years on file."""
        ...

    def search_chunks(
        self, query: Sequence[float], *, ciks: Sequence[str] | None = None, limit: int = 10
    ) -> list[ChunkHit]:
        """Nearest chunks by cosine similarity, from each company's latest section only."""
        ...


class InMemoryRiskStore:
    def __init__(self) -> None:
        self._sections: dict[int, StoredSection] = {}
        self._chunks: dict[int, StoredChunk] = {}
        self._factors: dict[int, StoredFactor] = {}
        self._diffs: dict[str, list[StoredDiff]] = {}
        self._next = {"section": 1, "chunk": 1, "factor": 1}

    def _id(self, kind: str) -> int:
        n = self._next[kind]
        self._next[kind] += 1
        return n

    def save_section(
        self, section: NewSection, factors: Sequence[NewFactor], chunks: Sequence[NewChunk]
    ) -> tuple[StoredSection, list[StoredFactor]]:
        for old in [s for s in self._sections.values() if s.accession == section.accession]:
            self._delete_section(old.id)
        stored = StoredSection(
            id=self._id("section"),
            cik=section.company.cik,
            ticker=section.company.ticker,
            accession=section.accession,
            fiscal_year=section.fiscal_year,
            report_date=section.report_date,
            source_url=section.source_url,
            method=section.method,
            split_method=section.split_method,
            confidence=section.confidence,
            needs_review=section.needs_review,
            word_count=section.word_count,
        )
        self._sections[stored.id] = stored
        chunk_ids: dict[int, list[int]] = {}
        for c in chunks:
            cid = self._id("chunk")
            self._chunks[cid] = StoredChunk(
                cid, stored.id, c.factor_ordinal, c.ordinal, c.heading, c.section_heading, c.text,
                c.token_count, list(c.embedding),
            )  # fmt: skip
            chunk_ids.setdefault(c.factor_ordinal, []).append(cid)
        out: list[StoredFactor] = []
        for f in factors:
            fid = self._id("factor")
            sf = StoredFactor(
                id=fid,
                section_id=stored.id,
                cik=stored.cik,
                fiscal_year=stored.fiscal_year,
                ordinal=f.ordinal,
                heading=f.heading,
                section_heading=f.section_heading,
                text=f.text,
                chunk_ids=chunk_ids.get(f.ordinal, []),
            )
            self._factors[fid] = sf
            out.append(sf)
        return stored, out

    def _delete_section(self, section_id: int) -> None:
        self._sections.pop(section_id, None)
        self._chunks = {k: v for k, v in self._chunks.items() if v.section_id != section_id}
        self._factors = {k: v for k, v in self._factors.items() if v.section_id != section_id}

    def save_classifications(
        self,
        results: dict[int, Classification],
        failures: dict[int, str],
        *,
        model: str,
        agent_run_id: int | None,
    ) -> None:
        for fid, c in results.items():
            self._factors[fid] = replace(
                self._factors[fid],
                status="classified",
                categories=[str(x) for x in c.categories],
                severity=c.severity,
                summary=c.summary,
                cited_chunk_ids=list(c.chunk_ids),
                model=model,
                agent_run_id=agent_run_id,
                failure=None,
            )
        for fid, reason in failures.items():
            self._factors[fid] = replace(self._factors[fid], status="failed", model=model, failure=reason)

    def sections(self, cik: str) -> list[StoredSection]:
        return sorted((s for s in self._sections.values() if s.cik == cik), key=lambda s: -s.fiscal_year)

    def section_by_id(self, section_id: int) -> StoredSection | None:
        return self._sections.get(section_id)

    def factors(self, section_id: int) -> list[StoredFactor]:
        return sorted(
            (f for f in self._factors.values() if f.section_id == section_id), key=lambda f: f.ordinal
        )

    def chunks(self, section_id: int) -> list[StoredChunk]:
        return sorted(
            (c for c in self._chunks.values() if c.section_id == section_id), key=lambda c: c.ordinal
        )

    def chunks_by_id(self, ids: Sequence[int]) -> list[StoredChunk]:
        return [self._chunks[i] for i in ids if i in self._chunks]

    def save_diffs(self, cik: str, from_year: int, to_year: int, rows: Sequence[DiffRow]) -> None:
        self._diffs[cik] = [
            StoredDiff(cik, from_year, to_year, r.status, r.from_id, r.to_id, r.similarity, r.text_similarity)
            for r in rows
        ]

    def diffs(self, cik: str) -> list[StoredDiff]:
        return list(self._diffs.get(cik, []))

    def search_chunks(
        self, query: Sequence[float], *, ciks: Sequence[str] | None = None, limit: int = 10
    ) -> list[ChunkHit]:
        latest: dict[str, StoredSection] = {}
        for s in self._sections.values():
            if (ciks is None or s.cik in ciks) and (
                s.cik not in latest or s.fiscal_year > latest[s.cik].fiscal_year
            ):
                latest[s.cik] = s
        by_id = {s.id: s for s in latest.values()}
        hits = [
            ChunkHit(c, by_id[c.section_id], cosine(query, c.embedding))
            for c in self._chunks.values()
            if c.section_id in by_id
        ]
        return sorted(hits, key=lambda h: -h.score)[:limit]
