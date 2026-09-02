"""phase 6: one queryable activity log, so "what is the agent doing" is a SELECT

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-01

CONCEPT: why a table rather than better log files.
  The container already logs richly, but a log line is only useful to whoever
  is holding a terminal at the time. Three separate incidents on 2026-09-01
  (a three-day auth outage, 57 dead-lettered messages, and a silent backlog)
  were all discoverable in `docker logs` and all went unnoticed, because
  nothing about the system invited a look. A table invites a look: it is the
  same SELECT every time, it survives `docker rm`, and it can be joined
  against runs and messages.

GOTCHA: the UNIQUE constraint on (message_id, node, at) is load-bearing, not
hygiene. The flush in app/activity.py writes the whole accumulated event list
each time it runs, and a message's list is seen by more than one persist node.
The constraint plus ON CONFLICT DO NOTHING is what makes that idempotent.
Dropping it would silently produce duplicate history.

GOTCHA: no foreign key on message_id, deliberately — see the model docstring.
Activity rows write from their own transaction and must never fail on a
referential race with the message they describe.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_activity",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("message_id", sa.Text(), nullable=True),
        sa.Column("level", sa.String(length=8), nullable=False),
        sa.Column("node", sa.String(length=32), nullable=False),
        sa.Column("event", sa.String(length=48), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("detail", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        # SET NULL rather than CASCADE: deleting a run must not erase the
        # record of what happened during it.
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(
            "message_id", "node", "at", name="uq_agent_activity_event"
        ),
    )
    op.create_index("ix_agent_activity_at", "agent_activity", ["at"])
    op.create_index("ix_agent_activity_node", "agent_activity", ["node"])
    op.create_index("ix_agent_activity_event", "agent_activity", ["event"])
    # WHY a separate DESC index: every human-facing query on this table is
    # "most recent first". Postgres can walk an ASC index backwards, so this
    # is a modest win rather than a necessity — but this table only grows and
    # the read pattern will not change.
    op.execute(
        "CREATE INDEX ix_agent_activity_at_desc ON agent_activity (at DESC)"
    )
    # Errors are a small fraction of rows and the one slice looked at under
    # pressure. A partial index keeps that lookup cheap no matter how large
    # the table gets.
    op.execute(
        "CREATE INDEX ix_agent_activity_errors ON agent_activity (at DESC) "
        "WHERE level IN ('error', 'warning')"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_activity_errors")
    op.execute("DROP INDEX IF EXISTS ix_agent_activity_at_desc")
    op.drop_index("ix_agent_activity_event", table_name="agent_activity")
    op.drop_index("ix_agent_activity_node", table_name="agent_activity")
    op.drop_index("ix_agent_activity_at", table_name="agent_activity")
    op.drop_table("agent_activity")
