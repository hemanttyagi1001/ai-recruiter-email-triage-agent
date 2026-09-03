"""
Is this a bounce — a mail server telling us a message never arrived?

What this module does and why it exists here: the agent sends mail, so the
agent receives non-delivery reports about the mail it sent. Those land in the
same INBOX the pipeline reads, and every one of them is a dead end — there is
no opportunity in it, no human on the other side, and nothing anyone will ever
act on. Left alone they accumulate: 30 of them in the observed mailbox, each
one paying for a classify call before being filed as not_recruitment.

CONCEPT: why this is a code rule and not a classifier category.
  The classifier already recognises these as not_recruitment, and that is
  enough to stop them being answered — but not enough to justify DELETING
  someone's mail. Trashing is irreversible on a 30-day timer, so the decision
  has to be one a reader can verify by looking at it, reproduce exactly, and
  test with fixtures. A model that is 99% right on 200 messages a day is
  wrong twice a day, and "wrong" here means a recruiter's mail in the bin.
  This is the same argument D11 makes for the decline rules and D14 for the
  outbound validator: a constraint that must not be violated lives in code.
  It also runs BEFORE classify, so a bounce costs no tokens at all.

CONCEPT: two independent signals, either one sufficient.
  1. THE SENDER. Bounces come from the mail system itself, and RFC 5321 §4.5.1
     reserves `postmaster` for exactly this. `mailer-daemon` is the near
     universal convention alongside it. No recruiter mails you from either.
  2. THE SUBJECT. Delivery reports announce themselves — "Delivery Status
     Notification (Failure)", "Undeliverable: <original subject>". Gmail's own
     web UI renders these as "Address not found", which is what a user sees
     and therefore what they will call it.
  Either alone is decisive, so they are OR'd. Requiring both would miss the
  Exchange servers that bounce from a real-looking address, and the Gmail
  reports whose From varies by relay.

GOTCHA: the subject markers are PHRASES, not words, and that is load-bearing.
  The observed inbox contains `shipment-tracking@amazon.in` with the subject
  "Out for delivery: ...". A bare `delivery` token would put a parcel
  notification in the bin. Every marker below is long enough that no ordinary
  mail carries it by accident, and the Amazon case is pinned as a negative
  test in tests/test_undeliverable.py.

GOTCHA: this module answers "is this a bounce", NOT "should it be trashed".
  Whether anything is deleted is decided by INBOX_CLEANUP_MODE and the kill
  switch, in app/pipeline/cleanup_node.py. Keeping the question separate from
  the consequence is what lets the detection be tested exhaustively without a
  Gmail client anywhere near it.
"""

from __future__ import annotations

import re

from app.gmail.parser import ParsedMessage

# Local parts reserved for the mail system. Matched on the LOCAL PART only,
# anchored, so `postmaster@anything` matches while a person called
# `ann.postmaster@corp.com` does not.
# WHY anchored where NO_REPLY_MARKERS in reply_target.py uses substrings: that
# module is deciding "would a reply be wasted", where a false positive costs
# one unsent reply. This one is deciding "should this be deleted", where a
# false positive costs an email. Same shape of test, different blast radius,
# so this one is strict.
DAEMON_LOCAL_PARTS: frozenset[str] = frozenset({
    "mailer-daemon",
    "mailerdaemon",
    "postmaster",
})

# Phrases that appear in the SUBJECT of a delivery report. Lowercased
# substring match — see the module GOTCHA on why these are all multi-word or
# otherwise unambiguous.
BOUNCE_SUBJECT_MARKERS: tuple[str, ...] = (
    "delivery status notification",
    "undeliverable",
    "address not found",
    "mail delivery failed",
    "mail delivery subsystem",
    "returned mail",
    "message not delivered",
    "delivery incomplete",
    "failure notice",
)

_LOCAL_PART_RE = re.compile(r"^([^@\s]+)@")


def _local_part(email: str | None) -> str | None:
    if not email:
        return None
    m = _LOCAL_PART_RE.match(email.strip().lower())
    return m.group(1) if m else None


def is_daemon_sender(from_email: str | None) -> bool:
    """True if the sender is the mail system rather than a person."""
    local = _local_part(from_email)
    return local is not None and local in DAEMON_LOCAL_PARTS


def has_bounce_subject(subject: str | None) -> bool:
    """True if the subject announces a delivery failure."""
    if not subject:
        return False
    lowered = subject.strip().lower()
    return any(marker in lowered for marker in BOUNCE_SUBJECT_MARKERS)


def is_undeliverable(parsed: ParsedMessage) -> bool:
    """True if this message is a non-delivery report.

    TRACE: called once per message from ingest_node, before classify and
    therefore before any LLM call. A True short-circuits the entire pipeline:
    the message is persisted with SKIPPED_UNDELIVERABLE and then handed to
    inbox_cleanup, which decides whether to trash it.
    """
    return is_daemon_sender(parsed.from_email) or has_bounce_subject(parsed.subject)
