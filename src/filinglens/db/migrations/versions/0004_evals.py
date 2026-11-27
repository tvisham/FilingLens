"""Phase 4: eval_runs, eval_results

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("suite", sa.String(20), nullable=False),
        sa.Column("git_sha", sa.String(64)),
        sa.Column("models", postgresql.JSONB, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("passed", sa.Boolean, nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
    )
    op.create_index("ix_eval_runs_suite_started", "eval_runs", ["suite", "started_at"])
    op.create_table(
        "eval_results",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "eval_run_id", sa.BigInteger, sa.ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("kind", sa.String(10), nullable=False),  # metric | case
        sa.Column("metric", sa.String(60), nullable=False),
        sa.Column("case_id", sa.String(200)),
        sa.Column("score", sa.Float),
        sa.Column("threshold", postgresql.JSONB),
        sa.Column("passed", sa.Boolean),  # NULL = skipped
        sa.Column("details", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("ix_eval_results_run_metric", "eval_results", ["eval_run_id", "metric"])


def downgrade() -> None:
    op.drop_table("eval_results")
    op.drop_table("eval_runs")
