"""
Activity log writer — the thing that makes `agent_activity` fill up.

What this module does and why it exists here: the agent already knew
everything about its own behaviour and kept almost none of it. Per-node
timings and outcomes accumulated in TriageState["events"] and were discarded
with the checkpoint; cycle boundaries and infra aborts existed only as lines
in a container log. This module writes both into one table so that "what is
the agent doing" is a SELECT rather than a request to go read logs.

Two entry points:
  record()       — one row, for something that just happened. Used for
                   process-level events (cycle start/finish, scope preflight,
                   infra abort) which have no NodeEvent to carry them.
  flush_events() — persist a whole accumulated NodeEvent list for one
                   message. Idempotent, so calling it twice is harmless.

CONCEPT: an audit log must never be able to break the thing it audits.
  Every write here is wrapped and swallowed on failure, exactly like
  app/dead_letter.py. If Postgres is down, the pipeline keeps running and
  loses log rows — the inverse (a healthy agent refusing to process mail
  because it could not write a log line) would be an absurd trade. The
  swallowed exception is itself logged, which is the fallback record.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert

from app.db.engine import session_scope
from app.db.models import AgentActivity

log = logging.getLogger(__name__)

# The `event` code used for a row flushed from a NodeEvent. A NodeEvent
# carries no code of its own — it has a node name and a prose outcome — so
# they all share one.
# WHY that is fine: the node name is the discriminator you actually want
# ("show me every auto_send"), and the specific codes below exist for the
# handful of moments that have no node. It also gives you the single most
# useful query on this table for free:
#     SELECT * FROM agent_activity WHERE event <> 'node_completed'
# which is the headline feed with the per-node detail filtered out.
NODE_COMPLETED = "node_completed"


def _write(rows: Sequence[dict[str, Any]]) -> int:
    """INSERT rows, ignoring any that collide with an existing event.

    CONCEPT: ON CONFLICT DO NOTHING as the idempotency mechanism.
      flush_events writes the whole accumulated list every time it is called,
      and the same early events reappear in later flushes for the same
      message. Rather than tracking a high-water mark in state (one more
      thing to get wrong across a checkpoint restore), the unique constraint
      on (message_id, node, at) rejects the repeats and the insert simply
      reports fewer rows written.
    GOTCHA: the returned count is best-effort and is NOT a success signal.
    psycopg reports rowcount as -1 for a multi-row INSERT ... ON CONFLICT,
    meaning "not available" rather than "nothing written" — verified against
    the installed driver. We normalise that to 0, so a 0 here means either
    "everything already existed" or "the driver declined to say". Treat this
    value as a debugging hint only; never branch on it. Whether the write
    succeeded is carried by the absence of an exception.
    """
    if not rows:
        return 0
    stmt = insert(AgentActivity).values(list(rows))
    stmt = stmt.on_conflict_do_nothing(constraint="uq_agent_activity_event")
    with session_scope() as s:
        result = s.execute(stmt)
        rowcount = result.rowcount
        return rowcount if rowcount is not None and rowcount > 0 else 0


def record(
    *,
    node: str,
    event: str,
    level: str = "info",
    outcome: str | None = None,
    duration_ms: int | None = None,
    message_id: str | None = None,
    run_id: UUID | None = None,
    detail: dict[str, Any] | None = None,
    at: datetime | None = None,
) -> None:
    """Write one activity row. Never raises.

    WHY its own session rather than joining the caller's: the most valuable
    rows describe failures, and a caller in the middle of a failure often
    holds a transaction that is already doomed. A fresh transaction means the
    record of the failure survives the failure. Same reasoning as
    app/dead_letter.py.
    """
    try:
        _write([{
            "at": at or datetime.now(timezone.utc),
            "run_id": run_id,
            "message_id": message_id,
            "level": level,
            "node": node,
            "event": event,
            "outcome": outcome,
            "duration_ms": duration_ms,
            "detail": detail,
        }])
    except Exception:
        # See the module docstring: log-and-swallow is the whole contract.
        log.exception(
            "failed to write agent_activity row (node=%s event=%s message_id=%s); "
            "continuing", node, event, message_id,
        )


def flush_events(
    events: Iterable[Any],
    *,
    message_id: str | None = None,
    run_id: UUID | None = None,
) -> int:
    """Persist an accumulated list of NodeEvents. Never raises.

    Returns a best-effort inserted-row count — see _write's GOTCHA; it is a
    debugging hint, not a success signal.

    TRACE: called once per message from ingest._process_one, immediately
    after graph.invoke() returns. At that point `final_state["events"]` holds
    one NodeEvent per node that ran, in execution order, including the nodes
    that ran before an interrupt. One call therefore captures the whole
    per-message trace without any node needing to know this table exists.

    GOTCHA: a message that pauses at the approval interrupt and is later
    resumed through the API produces a SECOND invoke whose events are not
    seen here. Those rows are missing until the API path calls this too —
    see D74. In draft mode almost nothing takes that path, which is why it
    was acceptable to land this without it.
    """
    rows: list[dict[str, Any]] = []
    for ev in events or []:
        try:
            rows.append({
                "at": ev.at,
                "run_id": run_id,
                "message_id": message_id,
                "level": "info",
                "node": ev.node,
                "event": NODE_COMPLETED,
                "outcome": ev.outcome,
                "duration_ms": ev.duration_ms,
                "detail": None,
            })
        except AttributeError:
            # A malformed event must not cost the rest of the trace.
            log.warning("skipping unreadable event %r for message_id=%s",
                        ev, message_id)
    try:
        return _write(rows)
    except Exception:
        log.exception(
            "failed to flush %d agent_activity events for message_id=%s; "
            "continuing", len(rows), message_id,
        )
        return 0


__all__ = ["NODE_COMPLETED", "flush_events", "record"]
