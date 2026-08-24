"""phase 2: draft approval columns + status backfill

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-19

NOTE: LangGraph's PostgresSaver creates its own tables (checkpoints,
checkpoint_writes, checkpoint_blobs) via `checkpointer.setup()` called at
first CLI/API boot. Those tables are NOT managed by Alembic — the
checkpointer owns its own schema. Dropping the DB drops them too;
recreating triggers setup() on next start.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- New Draft columns for the approval lifecycle ---
    op.add_column("drafts", sa.Column("approval_reason", sa.Text, nullable=True))
    op.add_column("drafts", sa.Column("gmail_draft_id", sa.String(64), nullable=True))
    op.add_column("drafts", sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True))

    # --- Backfill: Phase 1 drafts in pending_review become awaiting_approval ---
    # WHY as a data migration, not code: pre-Phase-2 rows shouldn't need the
    # app to be restarted with special "migration mode" to be visible in the
    # new /pending endpoint. Running the migration is the migration.
    # GOTCHA: this only covers `drafts.status`. The `messages` table's
    # status field for the same rows is likely `drafted` (Phase 1 terminal
    # state), which under Phase 2 semantics maps to `awaiting_approval`
    # since the draft was never sent to Gmail. We update both.
    op.execute(
        "UPDATE drafts SET status = 'awaiting_approval' WHERE status = 'pending_review'"
    )
    op.execute(
        "UPDATE messages SET status = 'awaiting_approval' "
        "WHERE status = 'drafted' AND message_id IN ("
        "  SELECT message_id FROM drafts WHERE status = 'awaiting_approval'"
        ")"
    )


def downgrade() -> None:
    # Reverse backfill (best effort — we can't distinguish Phase 1 rows
    # from Phase 2 rows that never left awaiting_approval).
    op.execute(
        "UPDATE messages SET status = 'drafted' WHERE status = 'awaiting_approval'"
    )
    op.execute(
        "UPDATE drafts SET status = 'pending_review' WHERE status = 'awaiting_approval'"
    )
    op.drop_column("drafts", "resolved_at")
    op.drop_column("drafts", "gmail_draft_id")
    op.drop_column("drafts", "approval_reason")
