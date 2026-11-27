"""Phase 5: review_queue, audit_log, agent_runs.guardrails

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("guardrails", postgresql.JSONB))
    op.add_column("agent_runs", sa.Column("repairs", sa.Integer, nullable=False, server_default="0"))

    op.create_table(
        "review_queue",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "agent_run_id", sa.BigInteger, sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("severity", sa.String(10), nullable=False),  # block | review
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("reviewer", sa.String(100)),
        sa.Column("decision_reason", sa.Text),
        sa.Column("edited_output", postgresql.JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'edited', 'rejected')", name="ck_review_queue_status"
        ),
    )
    op.create_index("ix_review_queue_status_created", "review_queue", ["status", "created_at"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("actor", sa.String(100), nullable=False),  # agent | system | user:<name>
        sa.Column("action", sa.String(60), nullable=False),
        sa.Column("entity", sa.String(40), nullable=False),
        sa.Column("entity_id", sa.BigInteger),
        sa.Column("payload", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_audit_log_entity", "audit_log", ["entity", "entity_id"])
    # Append-only: the application never updates or deletes audit rows.
    op.execute(
        """
        CREATE FUNCTION audit_log_append_only() RETURNS trigger AS $$
        BEGIN RAISE EXCEPTION 'audit_log is append-only'; END; $$ LANGUAGE plpgsql;
        CREATE TRIGGER audit_log_no_change BEFORE UPDATE OR DELETE ON audit_log
        FOR EACH ROW EXECUTE FUNCTION audit_log_append_only();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_log_no_change ON audit_log")
    op.execute("DROP FUNCTION IF EXISTS audit_log_append_only()")
    op.drop_table("audit_log")
    op.drop_table("review_queue")
    op.drop_column("agent_runs", "repairs")
    op.drop_column("agent_runs", "guardrails")
