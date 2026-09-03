"""
Ingest node — the graph's entry point. Runs first, does very little.

The CLI/API has already fetched the message from Gmail and parsed it (so
that thread_id = Message-ID is known before graph.invoke()). Ingest just:
  - emits the first NodeEvent (starts the audit trail)
  - does a belt-and-suspenders DB dedup check

The heavy lifting (fetch, parse) lives OUTSIDE the graph because
thread_id is derived from the parsed message. If we put fetch inside the
graph, we'd have to invoke without a stable thread_id and rename mid-run,
which LangGraph does not support.
"""

from __future__ import annotations

import time

from app.db.engine import session_scope
from app.db.models import Message, MessageStatus
from app.pipeline.state import TriageState, make_event
from app.rules.undeliverable import is_undeliverable


def ingest_node(state: TriageState) -> dict:
    # TRACE: called once per thread as the first node after START. Emits
    # one NodeEvent into the `events` list (appended via reducer). Does
    # NOT populate `final_status` on the happy path — that's for the
    # persist_* nodes.
    started = time.monotonic()
    parsed = state["parsed"]

    # Belt-and-suspenders: the CLI already checks by gmail_id and by
    # message_id before invoking. If we still find a duplicate here, it
    # means either a race between CLI runs or a resumed thread that
    # completed persist and then re-entered — both should short-circuit
    # cleanly.
    with session_scope() as s:
        existing = s.get(Message, parsed.message_id)
        already = existing is not None
        already_status = existing.status if existing else None

    duration_ms = int((time.monotonic() - started) * 1000)

    if already:
        return {
            "final_status": already_status,
            "events": [
                make_event(
                    "ingest",
                    outcome=f"duplicate: existing status={already_status}",
                    duration_ms=duration_ms,
                )
            ],
        }

    # D79: a bounce is recognised HERE, before classify, and never reaches an
    # LLM at all. Placing it in this node rather than adding an eighth
    # classifier category is what makes it free — the observed mailbox held 30
    # non-delivery reports, each of which had been paying for a classify call
    # to be told it was not recruitment mail.
    # TRACE: setting final_status routes to persist_terminal via
    # _route_after_ingest (the same edge the duplicate check above uses), which
    # writes the row and then hands to inbox_cleanup. Nothing is trashed until
    # that row exists — see D78 for why that order is not negotiable.
    if is_undeliverable(parsed):
        return {
            "final_status": MessageStatus.SKIPPED_UNDELIVERABLE,
            "events": [
                make_event(
                    "ingest",
                    outcome=(
                        f"non-delivery report from {parsed.from_email}; "
                        f"skipping before classify"
                    ),
                    duration_ms=duration_ms,
                )
            ],
        }

    return {
        "events": [
            make_event(
                "ingest",
                outcome=f"start message_id={parsed.message_id}",
                duration_ms=duration_ms,
            )
        ]
    }
