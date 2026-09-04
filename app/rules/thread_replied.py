"""
Who spoke last in this thread — us, or them?

What this module does and why it exists here: the agent drafts replies to mail
the operator has frequently already answered by hand. Nothing in the database
knows that. `already_replied` (D60) asks "have WE sent this address a reply",
and answers from the `drafts` table — which only records what the AGENT did.
A reply typed in Gmail leaves no row, so the agent happily drafts on top of a
conversation a human has already had. Measured on the live mailbox: 11 of 12
threads carrying an agent draft already had a sent reply in them.

CONCEPT: the rule, stated the way the operator stated it.
  Look at the LAST message in the thread.
    - last message is from the recruiter  -> they are waiting on us -> draft.
    - last message is from us             -> we already answered    -> skip.
  That is the whole rule, and its virtue is that it handles a running
  conversation without any special case. HR writes, we reply, HR writes again:
  the last message is theirs, so their follow-up gets an answer. HR writes and
  we reply: the last message is ours, so nothing further is drafted.

ALTERNATIVE considered and rejected: "is there any sent message newer than the
  message being processed". It gives the same answer in every ordinary case
  and is harder to hold in your head, because it reasons about a timestamp
  comparison rather than about a conversation. The one case where they differ
  is re-processing an OLD message in a thread that has since moved on, which
  the ingest guards already prevent.

CONCEPT: why the SENT label rather than comparing the From address.
  Gmail stamps SENT on every message this mailbox sent, so it is a fact about
  provenance rather than a string match. Comparing From to our own address
  would have to cope with aliases, plus-addressing, display-name formats and
  the send-as addresses Gmail permits — each a way to answer "did I write
  this" wrongly. The label cannot be spoofed by a sender.

GOTCHA: DRAFT-labelled messages are excluded before the last one is chosen.
  An unsent draft sits in the thread and would otherwise look like the final
  word, so the agent's own pending draft would suppress the very reply it is
  waiting to become. A draft is not an answer until it is sent.

GOTCHA: this module must not decide anything by itself. It reports a fact
  about a thread; whether that fact stops the pipeline is the graph's
  business. See app/pipeline/ingest_node.py.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

SENT_LABEL = "SENT"
DRAFT_LABEL = "DRAFT"


def _sortable_date(message: dict) -> int:
    """internalDate as an int, or 0 when Gmail omitted it.

    WHY not raise on a missing date: one malformed message must not decide
    that a thread is unanswered. Sorting it to the beginning means it can
    never be mistaken for the last word.
    """
    try:
        return int(message.get("internalDate") or 0)
    except (TypeError, ValueError):
        return 0


def we_spoke_last(messages: list[dict] | None) -> bool:
    """True if the most recent real message in the thread was sent by us.

    Takes raw Gmail message dicts (as returned by threads().get) rather than a
    client, so the whole rule is testable with fixtures and no network. That
    separation is the point: this is the function that decides whether a
    recruiter hears from us, and it should be exercisable exhaustively.
    """
    if not messages:
        # An empty or unreadable thread tells us nothing. Saying "we spoke
        # last" here would silently suppress a reply on no evidence.
        return False

    real = [m for m in messages if DRAFT_LABEL not in (m.get("labelIds") or [])]
    if not real:
        # A thread containing nothing but drafts: we have not answered.
        return False

    last = max(real, key=_sortable_date)
    return SENT_LABEL in (last.get("labelIds") or [])


def thread_already_answered(gmail: Any, thread_id: str | None) -> bool:
    """`we_spoke_last` for a thread id, with Gmail failures made harmless.

    TRACE: called once per new message from ingest_node, before classify, so a
    thread the operator already answered costs one metadata-only Gmail read
    and no LLM call at all.

    GOTCHA: fails OPEN, and deliberately, even though the whole feature exists
    to stop unwanted drafts. A Gmail error returning True would mean "assume
    answered", and every message during an outage would be silently dropped
    with no reply and no draft — the failure mode that loses a real
    opportunity with nothing to show for it. Returning False costs a draft the
    operator deletes. Same reasoning as D60, and the same as the rest of this
    codebase: guards on outbound actions fail open, the redactor fails closed.
    """
    if not thread_id:
        return False
    try:
        messages = gmail.thread_messages(thread_id)
    except Exception as exc:
        log.warning(
            "thread lookup failed for thread_id=%s (%s: %s); assuming NOT "
            "answered and continuing", thread_id, type(exc).__name__, exc,
        )
        return False
    return we_spoke_last(messages)
