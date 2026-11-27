"""SQLAlchemy models. Phase 1: companies, filings, financial_facts, cadence_snapshots.
Phase 2: agent_runs. Phase 3: risk_sections, chunks (pgvector), risk_factors, risk_diffs.
Phase 4: eval_runs, eval_results. Phase 5: review_queue, audit_log (append-only).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    cik: Mapped[str] = mapped_column(String(10), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(300))
    sector: Mapped[str] = mapped_column(String(50))
    sic: Mapped[str | None] = mapped_column(String(8))
    fiscal_year_end: Mapped[str | None] = mapped_column(String(4))  # MMDD
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Filing(Base):
    __tablename__ = "filings"
    __table_args__ = (Index("ix_filings_cik_form_date", "cik", "form", "filing_date"),)

    accession: Mapped[str] = mapped_column(String(25), primary_key=True)
    cik: Mapped[str] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"))
    form: Mapped[str] = mapped_column(String(20))
    is_amendment: Mapped[bool] = mapped_column(Boolean, default=False)
    filing_date: Mapped[date] = mapped_column(Date)
    report_date: Mapped[date | None] = mapped_column(Date)
    primary_document: Mapped[str | None] = mapped_column(String(300))
    url: Mapped[str | None] = mapped_column(Text)


class FinancialFact(Base):
    __tablename__ = "financial_facts"
    __table_args__ = (
        UniqueConstraint("cik", "metric", "period_start", "period_end", name="uq_financial_period"),
        CheckConstraint("fiscal_quarter BETWEEN 1 AND 4", name="ck_fiscal_quarter"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"), index=True)
    metric: Mapped[str] = mapped_column(String(20))
    concept: Mapped[str] = mapped_column(String(200))
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    duration_days: Mapped[int] = mapped_column(Integer)
    period_type: Mapped[str] = mapped_column(String(10))
    fiscal_year: Mapped[int | None] = mapped_column(Integer)
    fiscal_quarter: Mapped[int | None] = mapped_column(SmallInteger)
    value: Mapped[Decimal] = mapped_column(Numeric(24, 4))
    unit: Mapped[str] = mapped_column(String(10))
    accession: Mapped[str] = mapped_column(String(25))
    form: Mapped[str] = mapped_column(String(10))
    filed: Mapped[date] = mapped_column(Date)
    derived: Mapped[bool] = mapped_column(Boolean, default=False)
    derivation: Mapped[str | None] = mapped_column(Text)
    source_accessions: Mapped[list[str]] = mapped_column(ARRAY(String(25)))


class CadenceSnapshot(Base):
    __tablename__ = "cadence_snapshots"
    __table_args__ = (UniqueConstraint("cik", "window_end", "form", name="uq_cadence_snapshot"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"), index=True)
    window_start: Mapped[date] = mapped_column(Date)
    window_end: Mapped[date] = mapped_column(Date)
    form: Mapped[str] = mapped_column(String(20))
    count: Mapped[int] = mapped_column(Integer)
    amendments: Mapped[int] = mapped_column(Integer)
    dates: Mapped[list[date]] = mapped_column(ARRAY(Date))
    median_days_after_period_end: Mapped[float | None] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentRun(Base):
    """One agent invocation (Phase 2+): what it was asked, what it called, what it cost."""

    __tablename__ = "agent_runs"
    __table_args__ = (Index("ix_agent_runs_scenario_created", "scenario", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scenario: Mapped[str] = mapped_column(String(50))
    input: Mapped[dict[str, Any]] = mapped_column(JSONB)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(30))
    error: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(100))
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    num_turns: Mapped[int | None] = mapped_column(Integer)
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    session_id: Mapped[str | None] = mapped_column(String(100))
    guardrails: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    repairs: Mapped[int] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# -- Phase 3: risk factors ---------------------------------------------------------------

EMBEDDING_DIMENSIONS = 384  # BAAI/bge-small-en-v1.5


class RiskSection(Base):
    __tablename__ = "risk_sections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"), index=True)
    accession: Mapped[str] = mapped_column(ForeignKey("filings.accession", ondelete="CASCADE"), unique=True)
    fiscal_year: Mapped[int] = mapped_column(Integer)
    report_date: Mapped[date] = mapped_column(Date)
    source_url: Mapped[str] = mapped_column(Text)
    method: Mapped[str] = mapped_column(String(30))
    split_method: Mapped[str] = mapped_column(String(20))
    extraction_confidence: Mapped[float] = mapped_column(Float)
    needs_review: Mapped[bool] = mapped_column(Boolean)
    word_count: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("risk_section_id", "ordinal", name="uq_chunk_ordinal"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    risk_section_id: Mapped[int] = mapped_column(
        ForeignKey("risk_sections.id", ondelete="CASCADE"), index=True
    )
    factor_ordinal: Mapped[int] = mapped_column(Integer)
    ordinal: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(Text)
    section_heading: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))


class RiskFactorRow(Base):
    __tablename__ = "risk_factors"
    __table_args__ = (UniqueConstraint("risk_section_id", "ordinal", name="uq_risk_factor_ordinal"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    risk_section_id: Mapped[int] = mapped_column(
        ForeignKey("risk_sections.id", ondelete="CASCADE"), index=True
    )
    cik: Mapped[str] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"), index=True)
    fiscal_year: Mapped[int] = mapped_column(Integer)
    ordinal: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(Text)
    section_heading: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    chunk_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))
    status: Mapped[str] = mapped_column(String(20))
    categories: Mapped[list[str]] = mapped_column(ARRAY(String(30)))
    severity: Mapped[int | None] = mapped_column(SmallInteger)
    summary: Mapped[str | None] = mapped_column(Text)
    cited_chunk_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))
    model: Mapped[str | None] = mapped_column(String(200))
    agent_run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"))
    failure: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RiskDiff(Base):
    __tablename__ = "risk_diffs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cik: Mapped[str] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"), index=True)
    from_year: Mapped[int] = mapped_column(Integer)
    to_year: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    from_factor_id: Mapped[int | None] = mapped_column(ForeignKey("risk_factors.id", ondelete="CASCADE"))
    to_factor_id: Mapped[int | None] = mapped_column(ForeignKey("risk_factors.id", ondelete="CASCADE"))
    similarity: Mapped[float | None] = mapped_column(Float)
    text_similarity: Mapped[float | None] = mapped_column(Float)


# -- Phase 4: evals ----------------------------------------------------------------------


class EvalRun(Base):
    __tablename__ = "eval_runs"
    __table_args__ = (Index("ix_eval_runs_suite_started", "suite", "started_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    suite: Mapped[str] = mapped_column(String(20))
    git_sha: Mapped[str | None] = mapped_column(String(64))
    models: Mapped[dict[str, Any]] = mapped_column(JSONB)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    passed: Mapped[bool] = mapped_column(Boolean)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))


class EvalResult(Base):
    __tablename__ = "eval_results"
    __table_args__ = (Index("ix_eval_results_run_metric", "eval_run_id", "metric"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(10))
    metric: Mapped[str] = mapped_column(String(60))
    case_id: Mapped[str | None] = mapped_column(String(200))
    score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB)


# -- Phase 5: review queue and audit log --------------------------------------------------


class ReviewItem(Base):
    __tablename__ = "review_queue"
    __table_args__ = (Index("ix_review_queue_status_created", "status", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    reason: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20), server_default="pending")
    reviewer: Mapped[str | None] = mapped_column(String(100))
    decision_reason: Mapped[str | None] = mapped_column(Text)
    edited_output: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEntry(Base):
    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_log_entity", "entity", "entity_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(60))
    entity: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[int | None] = mapped_column(BigInteger)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# -- Phase 8: prices -----------------------------------------------------------------------


class PriceRow(Base):
    __tablename__ = "prices"

    cik: Mapped[str] = mapped_column(ForeignKey("companies.cik", ondelete="CASCADE"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    high: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    low: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    close: Mapped[Decimal] = mapped_column(Numeric(14, 4))
    volume: Mapped[int] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(120))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
