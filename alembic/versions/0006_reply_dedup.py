"""phase 6: record who each reply went to, so we never answer the same HR twice

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-22

CONCEPT: why a new column rather than deriving the address on demand.
  The recipient is not the sender. `resolve_reply_target` (D47/D59) picks the
  extracted recruiter_email over the From header, and can base64-decode a
  Naukri relay to get there — so `messages.from_email` is frequently NOT who
  we wrote to. Re-deriving it at query time would mean re-running that
  resolution over every historical row on every check, and would silently
  change answers whenever the resolver improves. Recording what we actually
  used makes "have we replied to this person" a lookup instead of a
  recomputation.

GOTCHA: nullable, because every row written before this migration has no
recorded recipient. The backfill in app/cli/backfill_reply_to.py reconstructs
them; rows it cannot resolve stay NULL and simply never match a dedup check,
which fails toward sending rather than toward silence.
"""

from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("drafts", sa.Column("reply_to_email", sa.Text(), nullable=True))
    # WHY lower(): email domains are case-insensitive and recruiters sign off
    # inconsistently — "Meghna.R@stellarhire.com" and "meghna.r@stellarhire.com"
    # are one person. The dedup query lowercases both sides, so the index has
    # to match that expression or Postgres will not use it.
    op.execute(
        "CREATE INDEX ix_drafts_reply_to_email_lower "
        "ON drafts (lower(reply_to_email))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_drafts_reply_to_email_lower")
    op.drop_column("drafts", "reply_to_email")
