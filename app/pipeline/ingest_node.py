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
from app.rules.thread_replied import thread_already_answered
from app.rules.undeliverable import is_undeliverable


def make_ingest_node(gmail=None):
    """Build the ingest node, optionally with Gmail access for the D81 check.

    CONCEPT: the three gates here are ordered by what they cost, cheapest
      first, and that ordering is the design rather than an accident.
        1. duplicate     — one indexed local SELECT.
        2. bounce (D79)  — pure string matching, no I/O at all.
        3. answered (D81)— one metadata-only Gmail read.
      Everything downstream of here costs money: classify and extract are two
      LLM calls before any of the later guards get a chance to speak. Spending
      one free Gmail read to avoid both is the trade, and on the observed
      mailbox it is a good one — 11 of 12 threads carrying an agent draft had
      already been answered by hand.

    WHY `gmail` is optional: tests that drive ingest directly do not need a
      client to exercise the duplicate and bounce paths, and passing None
      simply disables the thread check rather than requiring every caller to
      build a fake. Production always passes the real client, from
      build_graph.
    """

    def n(state: TriageState) -> dict:
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

        # D79: a bounce is recognised HERE, before classify, and never reaches
        # an LLM at all. Placing it in this node rather than adding an eighth
        # classifier category is what makes it free — the observed mailbox held
        # 30 non-delivery reports, each of which had been paying for a classify
        # call to be told it was not recruitment mail.
        # TRACE: setting final_status routes to persist_terminal via
        # _route_after_ingest (the same edge the duplicate check above uses),
        # which writes the row and then hands to inbox_cleanup. Nothing is
        # trashed until that row exists — see D78 for why that order matters.
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

        # D81: has this conversation already been answered? The rule is the
        # operator's: if the last message in the thread is one WE sent, they
        # have handled it and nothing further should be drafted; if the last
        # message is the recruiter's, they are waiting on us.
        # WHY here and not beside already_replied in the extract node: the
        # D60 check is free (a local SELECT) so it can afford to run late,
        # after classify and extract have already been paid for. This one
        # costs a Gmail round-trip but SAVES two LLM calls, so it belongs
        # before them. On the observed mailbox 11 of 12 threads carrying an
        # agent draft had already been answered by hand, so this is the common
        # case rather than the exception.
        # GOTCHA: fails open — see thread_already_answered. A Gmail outage
        # produces drafts to delete, never silent drops.
        if gmail is not None and thread_already_answered(gmail, parsed.thread_id):
            return {
                "final_status": MessageStatus.SKIPPED_THREAD_ANSWERED,
                "events": [
                    make_event(
                        "ingest",
                        outcome=(
                            "thread already answered — the last message in it "
                            "was sent by us; skipping before classify"
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

    return n


# Backwards-compatible module-level node with NO Gmail access, so the D81
# thread check is skipped. Production builds its own via make_ingest_node in
# build_graph; this exists for tests that drive ingest directly and care only
# about the duplicate and bounce paths.
ingest_node = make_ingest_node(None)
