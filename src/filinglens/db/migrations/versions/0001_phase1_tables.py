"""Phase 1 tables: companies, filings, financial_facts, cadence_snapshots

Revision ID: 0001
Revises:
Create Date: 2026-11-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("cik", sa.String(10), primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("sector", sa.String(50), nullable=False),
        sa.Column("sic", sa.String(8)),
        sa.Column("fiscal_year_end", sa.String(4)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_companies_ticker", "companies", ["ticker"], unique=True)

    op.create_table(
        "filings",
        sa.Column("accession", sa.String(25), primary_key=True),
        sa.Column("cik", sa.String(10), sa.ForeignKey("companies.cik", ondelete="CASCADE"), nullable=False),
        sa.Column("form", sa.String(20), nullable=False),
        sa.Column("is_amendment", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("filing_date", sa.Date, nullable=False),
        sa.Column("report_date", sa.Date),
        sa.Column("primary_document", sa.String(300)),
        sa.Column("url", sa.Text),
    )
    op.create_index("ix_filings_cik_form_date", "filings", ["cik", "form", "filing_date"])

    op.create_table(
        "financial_facts",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("cik", sa.String(10), sa.ForeignKey("companies.cik", ondelete="CASCADE"), nullable=False),
        sa.Column("metric", sa.String(20), nullable=False),
        sa.Column("concept", sa.String(200), nullable=False),
        sa.Column("period_start", sa.Date, nullable=False),
        sa.Column("period_end", sa.Date, nullable=False),
        sa.Column("duration_days", sa.Integer, nullable=False),
        sa.Column("period_type", sa.String(10), nullable=False),
        sa.Column("fiscal_year", sa.Integer),
        sa.Column("fiscal_quarter", sa.SmallInteger),
        sa.Column("value", sa.Numeric(24, 4), nullable=False),
        sa.Column("unit", sa.String(10), nullable=False),
        sa.Column("accession", sa.String(25), nullable=False),
        sa.Column("form", sa.String(10), nullable=False),
        sa.Column("filed", sa.Date, nullable=False),
        sa.Column("derived", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("derivation", sa.Text),
        sa.Column("source_accessions", postgresql.ARRAY(sa.String(25)), nullable=False),
        sa.UniqueConstraint("cik", "metric", "period_start", "period_end", name="uq_financial_period"),
        sa.CheckConstraint("fiscal_quarter BETWEEN 1 AND 4", name="ck_fiscal_quarter"),
    )
    op.create_index("ix_financial_facts_cik", "financial_facts", ["cik"])

    op.create_table(
        "cadence_snapshots",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("cik", sa.String(10), sa.ForeignKey("companies.cik", ondelete="CASCADE"), nullable=False),
        sa.Column("window_start", sa.Date, nullable=False),
        sa.Column("window_end", sa.Date, nullable=False),
        sa.Column("form", sa.String(20), nullable=False),
        sa.Column("count", sa.Integer, nullable=False),
        sa.Column("amendments", sa.Integer, nullable=False),
        sa.Column("dates", postgresql.ARRAY(sa.Date), nullable=False),
        sa.Column("median_days_after_period_end", sa.Float),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("cik", "window_end", "form", name="uq_cadence_snapshot"),
    )
    op.create_index("ix_cadence_snapshots_cik", "cadence_snapshots", ["cik"])


def downgrade() -> None:
    op.drop_table("cadence_snapshots")
    op.drop_table("financial_facts")
    op.drop_table("filings")
    op.drop_table("companies")
