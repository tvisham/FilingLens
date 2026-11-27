"""Phase 2: agent_runs

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("scenario", sa.String(50), nullable=False),
        sa.Column("input", postgresql.JSONB, nullable=False),
        sa.Column("output", postgresql.JSONB),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("model", sa.String(100)),
        sa.Column("tokens_in", sa.Integer),
        sa.Column("tokens_out", sa.Integer),
        sa.Column("cost_usd", sa.Numeric(12, 6)),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("num_turns", sa.Integer),
        sa.Column("tool_calls", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("session_id", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_agent_runs_scenario_created", "agent_runs", ["scenario", "created_at"])


def downgrade() -> None:
    op.drop_table("agent_runs")
