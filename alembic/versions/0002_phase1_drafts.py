"""phase 1: drafts table + run counters

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Runs: additional counters and score-node token accounting ---
    op.add_column("runs", sa.Column("score_tokens_in", sa.BigInteger, nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("score_tokens_out", sa.BigInteger, nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("rules_declined", sa.Integer, nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("messages_scored", sa.Integer, nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("messages_needs_review", sa.Integer, nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("drafts_created", sa.Integer, nullable=False, server_default="0"))
    op.add_column("runs", sa.Column("drafts_quarantined", sa.Integer, nullable=False, server_default="0"))

    # --- Drafts table ---
    op.create_table(
        "drafts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "message_id",
            sa.Text,
            sa.ForeignKey("messages.message_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "opportunity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("opportunities.id"),
            nullable=True,
        ),
        sa.Column("draft_type", sa.String(16), nullable=False),
        sa.Column("body_text", sa.Text, nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("quarantine_reason", sa.Text, nullable=True),
        sa.Column("rule_name", sa.String(64), nullable=True),
        sa.Column("rule_reason", sa.Text, nullable=True),
        sa.Column("fit_score", sa.Integer, nullable=True),
        sa.Column("fit_rationale", sa.Text, nullable=True),
        sa.Column("fit_uncertain", sa.Boolean, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_drafts_message_id", "drafts", ["message_id"], unique=True)
    op.create_index("ix_drafts_status", "drafts", ["status"])


def downgrade() -> None:
    op.drop_index("ix_drafts_status", table_name="drafts")
    op.drop_index("ix_drafts_message_id", table_name="drafts")
    op.drop_table("drafts")
    op.drop_column("runs", "drafts_quarantined")
    op.drop_column("runs", "drafts_created")
    op.drop_column("runs", "messages_needs_review")
    op.drop_column("runs", "messages_scored")
    op.drop_column("runs", "rules_declined")
    op.drop_column("runs", "score_tokens_out")
    op.drop_column("runs", "score_tokens_in")
