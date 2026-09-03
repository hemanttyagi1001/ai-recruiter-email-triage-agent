"""
The one node in this pipeline that removes something from the mailbox.

What this module does and why it exists here: the agent sends mail, so it
receives non-delivery reports about mail that failed. `ingest_node` detects
them deterministically (app/rules/undeliverable.py) and `persist_terminal`
records them; this node is what actually takes them out of the inbox. It runs
last, after the database row exists, and it is the only place in the codebase
that mutates a message the agent did not itself create.

CONCEPT: why this is a node rather than three lines inside persist_terminal.
  Persisting and mutating a third-party system are different kinds of act with
  different failure semantics, and folding them together is what caused the
  duplicate-draft incident (D78). Keeping the Gmail call in its own node makes
  the ORDER visible in the graph — persist_terminal, THEN cleanup — instead of
  buried inside a function whose name says it writes to the database. A reader
  looking at the graph edges can see that nothing is trashed before it is
  recorded.

CONCEPT: three gates, all of which must open.
  1. The message must be a bounce, decided in code by a module with no Gmail
     access and no LLM anywhere near it.
  2. INBOX_CLEANUP_MODE must be `trash`. `off` and `dry_run` both reach this
     node and both leave the mailbox untouched.
  3. The kill switch must be off.
  GOTCHA on the third: D49 exempts draft-mode from the kill switch, arguing a
  Gmail draft is not outbound mail and so nothing leaves the building. That
  argument does not transfer. Trashing is not "less outbound than sending" —
  it is a different axis entirely, an irreversible-in-30-days mutation of mail
  the operator owns. When someone pulls the switch because the agent is
  misbehaving, "it kept deleting my mail, but it never sent anything" is not a
  defensible outcome. So the switch gates this too. See D79.

TRACE at runtime, for one bounce:
  ingest (is_undeliverable → sets final_status)
    → persist_terminal (writes the messages row, status=skipped_undeliverable)
      → inbox_cleanup (this module; trashes, or logs, or does nothing)
        → END
  No LLM call happens anywhere on that path.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.db.models import MessageStatus
from app.kill_switch import is_send_halted
from app.pipeline.state import TriageState, make_event

log = logging.getLogger(__name__)


def make_inbox_cleanup_node(gmail):
    """Build the cleanup node, closing over the Gmail client.

    WHY a factory like the other I/O nodes (act, auto_send): it keeps the
    Gmail client injectable, so tests drive this with a fake that records
    calls instead of talking to Google. A node that reached for a module-level
    client would make "did we trash exactly the right message, and nothing
    else" untestable, which for this particular node is the only question
    worth asking.
    """

    def n(state: TriageState) -> dict:
        parsed = state["parsed"]

        # GOTCHA: this node is on the graph edge out of persist_terminal, which
        # every skipped message traverses — wrong category, no reply target,
        # already replied, extraction failed. Only bounces may be trashed, so
        # the status is re-checked here rather than trusted from the routing.
        # A routing bug must not become a deletion bug.
        if state.get("final_status") != MessageStatus.SKIPPED_UNDELIVERABLE:
            return {}

        mode = settings.inbox_cleanup_mode

        if mode == "off":
            return {
                "events": [make_event(
                    "inbox_cleanup", outcome="disabled (INBOX_CLEANUP_MODE=off)"
                )],
            }

        if mode == "dry_run":
            # WHY log the subject and sender: the entire purpose of dry_run is
            # letting a human check the detector against real mail before
            # arming it. A log line saying "would trash 1 message" proves
            # nothing; naming what it would have taken is reviewable.
            log.info(
                "DRY RUN: would trash gmail_id=%s from=%r subject=%r",
                parsed.gmail_id, parsed.from_email, parsed.subject,
            )
            return {
                "events": [make_event(
                    "inbox_cleanup", outcome="dry_run: would trash"
                )],
            }

        # TRACE: read fresh from the database on every message, never cached,
        # for the same reason auto_send does it — the switch must take effect
        # mid-run (D36). An operator flipping it should not have to wait for
        # the next cycle to stop the agent touching their mail.
        if is_send_halted():
            log.warning(
                "inbox_cleanup halted by kill switch; leaving gmail_id=%s in "
                "the inbox", parsed.gmail_id,
            )
            return {
                "events": [make_event(
                    "inbox_cleanup", outcome="halted by kill switch"
                )],
            }

        trashed = gmail.trash_message(parsed.gmail_id)
        # trash_message never raises — it returns False and logs. So a Gmail
        # outage or a token missing gmail.modify degrades to "the bounce is
        # still in the inbox", which the next cycle will not retry (the
        # messages row already exists, so the gmail_id guard skips it). That
        # is deliberate: an untidy inbox is a smaller problem than a retry
        # loop against an API that is refusing us.
        return {
            "events": [make_event(
                "inbox_cleanup",
                outcome="trashed" if trashed else "trash failed; left in inbox",
            )],
        }

    return n
