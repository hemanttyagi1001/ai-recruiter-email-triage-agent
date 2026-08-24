"""phase 5: dead_letters, system_flags, drafts.auto_actioned, AUTO_SENT statuses

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-20

CONCEPT: what this migration wires up for the operational safety layer.
  Phase 5 turns on limited autonomy (rule-based decline auto-send) and
  puts three defenses in place:
    - dead_letters: durable record of infra failures that exhausted the
      retry policy. Prevents "silently retried forever" or "silently
      dropped on the floor" — the human digest surfaces these.
    - system_flags: single-row-per-key config the pipeline consults per
      send. Today one key, sends_halted; tomorrow could hold rate caps,
      per-recruiter mutes, etc.
    - drafts.auto_actioned: distinguishes an autonomous send from one
      a human approved. The digest and any post-hoc audit both need to
      count these separately, and doing it on a column rather than by
      inferring from "no approval_reason set" avoids a load-bearing
      absence-of-value inference.

CONCEPT: why system_flags is a table, not a config field or env var.
  A config file (candidate.toml) is loaded at process start; changing it
  requires a restart to take effect. An env var is the same. The whole
  point of the kill switch is "flip it and the next send stops" without
  redeploying. A DB row can be flipped by `psql -c "UPDATE ..."` and the
  next `is_send_halted()` call reads the new value. The check location
  matters as much as the storage: `app/kill_switch.py` reads per-send,
  never cached. See D36.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- 1. dead_letters ---
    # WHY message_id is nullable: some failures happen before we've
    # parsed the message (e.g. gmail.list_message_ids returns 401). No
    # message row exists yet, but we still want the failure recorded so
    # the digest can surface "your run died at listing time."
    # WHY error_details is JSONB, not TEXT: attempt count, elapsed_ms,
    # and any request-id we can pull are structured; JSONB gives us
    # queryability (find all rate-limit failures across a week without
    # regex-scanning a text column).
    op.create_table(
        "dead_letters",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "message_id", sa.Text,
            sa.ForeignKey("messages.message_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "run_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("node", sa.String(64), nullable=False),
        sa.Column("error_class", sa.String(128), nullable=False),
        sa.Column("error_message", sa.Text, nullable=False),
        sa.Column(
            "error_details", postgresql.JSONB, nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "occurred_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
    )
    # WHY index on occurred_at: the digest query filters "last 24h"; a
    # scan without this index becomes O(all failures ever) as the table
    # grows. Cheap b-tree, worth it for a table that's read once a day.
    op.create_index(
        "ix_dead_letters_occurred_at", "dead_letters", ["occurred_at"],
    )

    # --- 2. system_flags ---
    # WHY the value column is BOOLEAN not JSONB: today the only key
    # (sends_halted) is a boolean. If future flags need richer values,
    # add another column (value_json JSONB) rather than converting this
    # one — keeps the common boolean path cheap and simple.
    # WHY the seed row is INSERTed in the same migration: an application
    # that reads sends_halted must not have to handle "row missing" as a
    # separate case. Seeding at migration time means the query always
    # returns exactly one row.
    op.create_table(
        "system_flags",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.Boolean, nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
        sa.Column("updated_by", sa.String(128), nullable=True),
    )
    op.execute(
        "INSERT INTO system_flags (key, value, updated_by) "
        "VALUES ('sends_halted', false, 'migration_0005')"
    )

    # --- 3. drafts.auto_actioned ---
    # WHY server_default false: existing rows (Phase 2 drafts) were all
    # human-approved by definition — no auto path existed. Default false
    # backfills correctly with no data migration.
    op.add_column(
        "drafts",
        sa.Column(
            "auto_actioned", sa.Boolean,
            nullable=False, server_default=sa.false(),
        ),
    )

    # No enum-alter needed for the new AUTO_SENT status values — enum
    # columns are TEXT (see D10), so a new value is a code-only change
    # in app.db.models.MessageStatus / DraftStatus.


def downgrade() -> None:
    op.drop_column("drafts", "auto_actioned")
    op.drop_table("system_flags")
    op.drop_index("ix_dead_letters_occurred_at", table_name="dead_letters")
    op.drop_table("dead_letters")
