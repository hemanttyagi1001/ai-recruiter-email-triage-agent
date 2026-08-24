"""
One reply per recruiter (D60).

The failure this prevents is not a crash. It is a recruiter who mails about
three roles receiving three near-identical generated emails — which reads
worse than no reply at all, and under AUTO_SEND_MODE=on happens without
anyone noticing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.db.models import Draft, DraftStatus, DraftType, Message, MessageStatus, Run, RunStatus
from app.rules.already_replied import already_replied


def _seed(db, reply_to: str, status: str, gmail_id: str = "g1"):
    run = Run(status=RunStatus.RUNNING)
    db.add(run)
    db.flush()
    mid = f"<{gmail_id}@x>"
    db.add(Message(
        message_id=mid, gmail_id=gmail_id, thread_id="t",
        from_email="whatever@sender.com", from_name=None, subject="s",
        received_at=datetime.now(timezone.utc), body_text="b", raw_headers={},
        run_id=run.id, status=MessageStatus.AUTO_SENT,
    ))
    # GOTCHA: flush before the Draft. drafts.message_id is a FK on a NATURAL
    # key with no ORM relationship configured, so SQLAlchemy cannot infer the
    # insert order and emits the child first.
    db.flush()
    db.add(Draft(
        message_id=mid, draft_type=DraftType.INTERESTED, body_text="body",
        status=status, reply_to_email=reply_to,
    ))
    db.commit()


def test_sent_reply_blocks_a_second_one(committed_db):
    _seed(committed_db, "ritika.s@brightpathtech.in", DraftStatus.AUTO_SENT)
    assert already_replied("ritika.s@brightpathtech.in") is True


def test_unsent_draft_does_not_block(committed_db):
    """D60 dedups on SENT mail only, by operator choice.

    A draft sitting in Gmail was never delivered — the recruiter has heard
    nothing, so they should still get a reply.
    """
    _seed(committed_db, "hr28@quickapply.com", DraftStatus.SENT_TO_GMAIL_DRAFTS)
    assert already_replied("hr28@quickapply.com") is False


def test_awaiting_approval_does_not_block(committed_db):
    _seed(committed_db, "hr@x.com", DraftStatus.AWAITING_APPROVAL)
    assert already_replied("hr@x.com") is False


def test_matching_is_case_insensitive(committed_db):
    """Recruiters sign off inconsistently.

    `Meghna.R@stellarhire.com` and `meghna.r@stellarhire.com` are one mailbox —
    email domains are case-insensitive, and treating them as two people would
    send that person a second copy.
    """
    _seed(committed_db, "Meghna.R@stellarhire.com", DraftStatus.AUTO_SENT)
    assert already_replied("meghna.r@stellarhire.com") is True
    assert already_replied("MEGHNA.R@STELLARHIRE.COM") is True


def test_whitespace_is_ignored(committed_db):
    _seed(committed_db, "hr@x.com", DraftStatus.AUTO_SENT)
    assert already_replied("  hr@x.com  ") is True


def test_unknown_address_is_not_blocked(committed_db):
    _seed(committed_db, "someone@else.com", DraftStatus.AUTO_SENT)
    assert already_replied("brand.new@recruiter.com") is False


def test_none_and_empty_are_never_blocked():
    assert already_replied(None) is False
    assert already_replied("") is False


def test_db_failure_fails_open(monkeypatch):
    """A DB blip must look like "not replied", never like "already handled".

    Failing closed would make a transient outage silently drop a real
    recruiter's first contact, with no row and no trace. A duplicate reply is
    embarrassing; a dropped one costs an opportunity.
    """
    import app.rules.already_replied as mod

    def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mod, "session_scope", boom)
    assert already_replied("hr@x.com") is False


def test_dedups_across_relay_and_direct(committed_db):
    """The whole point of deduping on the RESOLVED address.

    The same recruiter reaches you once directly and once through a Naukri
    relay. Two different senders, one mailbox — and because reply_to is what
    gets recorded, the second message sees the first reply.
    """
    _seed(committed_db, "asha@midwestconsultants.net", DraftStatus.AUTO_SENT)
    # The relay message resolves (via D59 decode) to the same address.
    assert already_replied("asha@midwestconsultants.net") is True
