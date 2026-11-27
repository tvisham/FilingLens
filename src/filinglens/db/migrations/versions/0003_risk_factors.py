"""Phase 3: risk_sections, chunks (pgvector), risk_factors, risk_diffs

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-26

Needs the pgvector extension. Neon and the `pgvector/pgvector` image ship it; on Neon the
database owner can run `CREATE EXTENSION`, which this migration does if it is missing.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIMENSIONS = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "risk_sections",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("cik", sa.String(10), sa.ForeignKey("companies.cik", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "accession",
            sa.String(25),
            sa.ForeignKey("filings.accession", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("fiscal_year", sa.Integer, nullable=False),
        sa.Column("report_date", sa.Date, nullable=False),
        sa.Column("source_url", sa.Text, nullable=False),
        sa.Column("method", sa.String(30), nullable=False),
        sa.Column("split_method", sa.String(20), nullable=False),
        sa.Column("extraction_confidence", sa.Float, nullable=False),
        sa.Column("needs_review", sa.Boolean, nullable=False),
        sa.Column("word_count", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding_model", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_risk_sections_cik", "risk_sections", ["cik"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "risk_section_id",
            sa.BigInteger,
            sa.ForeignKey("risk_sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("factor_ordinal", sa.Integer, nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("heading", sa.Text),
        sa.Column("section_heading", sa.Text),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(DIMENSIONS), nullable=False),
        sa.UniqueConstraint("risk_section_id", "ordinal", name="uq_chunk_ordinal"),
    )
    op.create_index("ix_chunks_risk_section_id", "chunks", ["risk_section_id"])
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "risk_factors",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "risk_section_id",
            sa.BigInteger,
            sa.ForeignKey("risk_sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cik", sa.String(10), sa.ForeignKey("companies.cik", ondelete="CASCADE"), nullable=False),
        sa.Column("fiscal_year", sa.Integer, nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("heading", sa.Text),
        sa.Column("section_heading", sa.Text),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("chunk_ids", postgresql.ARRAY(sa.BigInteger), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("categories", postgresql.ARRAY(sa.String(30)), nullable=False, server_default="{}"),
        sa.Column("severity", sa.SmallInteger),
        sa.Column("summary", sa.Text),
        sa.Column("cited_chunk_ids", postgresql.ARRAY(sa.BigInteger), nullable=False, server_default="{}"),
        sa.Column("model", sa.String(200)),
        sa.Column("agent_run_id", sa.BigInteger, sa.ForeignKey("agent_runs.id", ondelete="SET NULL")),
        sa.Column("failure", sa.Text),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("risk_section_id", "ordinal", name="uq_risk_factor_ordinal"),
        sa.CheckConstraint("severity IS NULL OR severity BETWEEN 1 AND 5", name="ck_risk_factor_severity"),
    )
    op.create_index("ix_risk_factors_risk_section_id", "risk_factors", ["risk_section_id"])
    op.create_index("ix_risk_factors_cik", "risk_factors", ["cik"])

    op.create_table(
        "risk_diffs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("cik", sa.String(10), sa.ForeignKey("companies.cik", ondelete="CASCADE"), nullable=False),
        sa.Column("from_year", sa.Integer, nullable=False),
        sa.Column("to_year", sa.Integer, nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("from_factor_id", sa.BigInteger, sa.ForeignKey("risk_factors.id", ondelete="CASCADE")),
        sa.Column("to_factor_id", sa.BigInteger, sa.ForeignKey("risk_factors.id", ondelete="CASCADE")),
        sa.Column("similarity", sa.Float),
        sa.Column("text_similarity", sa.Float),
    )
    op.create_index("ix_risk_diffs_cik", "risk_diffs", ["cik"])


def downgrade() -> None:
    op.drop_table("risk_diffs")
    op.drop_table("risk_factors")
    op.drop_table("chunks")
    op.drop_table("risk_sections")
    # The vector extension is left installed: other database objects may use it.
