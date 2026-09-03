"""
Tests for bounce detection and the one node that removes mail (D79).

These are gating tests for an irreversible action, so they are written in the
safe direction throughout: the negative cases matter more than the positive
ones. A detector that misses a bounce costs an untidy inbox; a detector that
fires on a recruiter costs an opportunity, and a 30-day window to notice.

The fixtures below are drawn from the real observed mailbox — including
`shipment-tracking@amazon.in` with the subject "Out for delivery: ...", which
is the exact message a naive `delivery` substring match would have binned.

All offline. No Gmail, no network, no LLM.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.db.models import MessageStatus
from app.pipeline.cleanup_node import make_inbox_cleanup_node
from app.rules.undeliverable import (
    has_bounce_subject,
    is_daemon_sender,
    is_undeliverable,
)


# --- Detection: the positives -----------------------------------------------


@pytest.mark.parametrize(
    "from_email",
    [
        "mailer-daemon@googlemail.com",
        "MAILER-DAEMON@googlemail.com",
        "postmaster@navbackoffice.com",
        "postmaster@organicut.onmicrosoft.com",
        "  mailer-daemon@gmail.com  ",
    ],
)
def test_daemon_senders_are_recognised(from_email, parsed_factory):
    assert is_undeliverable(parsed_factory(from_email=from_email))


@pytest.mark.parametrize(
    "subject",
    [
        "Delivery Status Notification (Failure)",
        "Undeliverable: Experienced AI/ML Engineer (GenAI, LLM, RAG)",
        "Address not found",
        "Mail delivery failed: returning message to sender",
        "Returned mail: see transcript for details",
        "failure notice",
    ],
)
def test_bounce_subjects_are_recognised(subject, parsed_factory):
    """From a perfectly ordinary-looking sender — the subject alone decides."""
    assert is_undeliverable(
        parsed_factory(from_email="relay@somecorp.com", subject=subject)
    )


def test_either_signal_alone_is_enough(parsed_factory):
    """The two checks are OR'd, and each is tested without the other."""
    assert is_daemon_sender("mailer-daemon@googlemail.com")
    assert not has_bounce_subject("A recruiter reaches out")
    assert is_undeliverable(
        parsed_factory(
            from_email="mailer-daemon@googlemail.com",
            subject="A recruiter reaches out",
        )
    )


# --- Detection: the negatives, which are the ones that matter ---------------


def test_a_parcel_notification_is_not_a_bounce(parsed_factory):
    """Regression for the exact message that made the markers phrases.

    `shipment-tracking@amazon.in` / "Out for delivery: ..." is real mail from
    the observed inbox. A `delivery` substring match would trash it.
    """
    assert not is_undeliverable(
        parsed_factory(
            from_email="shipment-tracking@amazon.in",
            subject='Out for delivery: "Manforce 3 in 1 Condoms..."',
        )
    )


@pytest.mark.parametrize(
    "subject",
    [
        "Delivery Manager role at Acme, 32 LPA",
        "Urgent: Senior Delivery Lead opening",
        "Out for delivery: your order",
        "Re: Following up on the AI/ML position",
        "We could not find a match for your profile",
    ],
)
def test_ordinary_mail_mentioning_delivery_is_not_a_bounce(subject, parsed_factory):
    """'Delivery Manager' is a real job title. This is why no marker is a
    bare word."""
    assert not is_undeliverable(
        parsed_factory(from_email="asha@midwestconsultants.net", subject=subject)
    )


@pytest.mark.parametrize(
    "from_email",
    [
        "ann.postmaster@corp.com",
        "postmaster.general@corp.com",
        "mailer-daemon.recruiting@corp.com",
    ],
)
def test_a_person_whose_address_contains_a_daemon_word_is_not_a_bounce(
    from_email, parsed_factory
):
    """The local part is matched WHOLE and anchored, unlike the substring
    matching in reply_target.NO_REPLY_MARKERS. That module decides whether a
    reply would be wasted; this one decides whether mail gets deleted, so it
    is strict where the other is generous."""
    assert not is_undeliverable(
        parsed_factory(from_email=from_email, subject="Opening for AI/ML Engineer")
    )


def test_a_missing_subject_does_not_crash_or_match(parsed_factory):
    assert not has_bounce_subject(None)
    assert not has_bounce_subject("")
    assert not is_undeliverable(
        parsed_factory(from_email="asha@corp.com", subject="")
    )


# --- The cleanup node: three gates, all must open ---------------------------


class FakeGmail:
    """Records trash calls instead of making them."""

    def __init__(self, succeed: bool = True):
        self.trashed: list[str] = []
        self._succeed = succeed

    def trash_message(self, gmail_id: str) -> bool:
        self.trashed.append(gmail_id)
        return self._succeed


def _state(parsed, status=MessageStatus.SKIPPED_UNDELIVERABLE):
    return {"parsed": parsed, "final_status": status}


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(settings, "inbox_cleanup_mode", "trash")
    monkeypatch.setattr(
        "app.pipeline.cleanup_node.is_send_halted", lambda: False
    )


def test_trashes_a_bounce_when_armed(parsed_factory, armed):
    gmail = FakeGmail()
    node = make_inbox_cleanup_node(gmail)
    out = node(_state(parsed_factory(gmail_id="g-bounce")))
    assert gmail.trashed == ["g-bounce"]
    assert out["events"][0].outcome == "trashed"


@pytest.mark.parametrize(
    "status",
    [
        MessageStatus.SKIPPED_WRONG_CATEGORY,
        MessageStatus.SKIPPED_NO_REPLY_TARGET,
        MessageStatus.SKIPPED_ALREADY_REPLIED,
        MessageStatus.EXTRACTION_FAILED,
        MessageStatus.AUTO_SENT,
    ],
)
def test_never_trashes_anything_that_is_not_a_bounce(status, parsed_factory, armed):
    """THE test. Every skipped message traverses persist_terminal, so this node
    sits on an edge that all of them can reach. A routing bug must not become a
    deletion bug, which is why the node re-checks the status the router
    already branched on."""
    gmail = FakeGmail()
    node = make_inbox_cleanup_node(gmail)
    node(_state(parsed_factory(gmail_id="g-keep"), status=status))
    assert gmail.trashed == []


def test_off_mode_touches_nothing(parsed_factory, monkeypatch):
    monkeypatch.setattr(settings, "inbox_cleanup_mode", "off")
    gmail = FakeGmail()
    out = make_inbox_cleanup_node(gmail)(_state(parsed_factory()))
    assert gmail.trashed == []
    assert "disabled" in out["events"][0].outcome


def test_dry_run_reports_without_touching_gmail(parsed_factory, monkeypatch):
    monkeypatch.setattr(settings, "inbox_cleanup_mode", "dry_run")
    gmail = FakeGmail()
    out = make_inbox_cleanup_node(gmail)(_state(parsed_factory()))
    assert gmail.trashed == []
    assert "dry_run" in out["events"][0].outcome


def test_the_kill_switch_stops_deletion(parsed_factory, monkeypatch):
    """D79 departs from D49 here, deliberately.

    Draft mode is exempt from the kill switch because a draft is not outbound
    mail. Trashing is not less consequential than sending — it is irreversible
    on a 30-day timer against mail the operator owns. "It kept deleting my
    mail, but it never sent anything" is not a defensible outcome for a switch
    someone pulled in a panic.
    """
    monkeypatch.setattr(settings, "inbox_cleanup_mode", "trash")
    monkeypatch.setattr(
        "app.pipeline.cleanup_node.is_send_halted", lambda: True
    )
    gmail = FakeGmail()
    out = make_inbox_cleanup_node(gmail)(_state(parsed_factory()))
    assert gmail.trashed == []
    assert "halted" in out["events"][0].outcome


def test_a_gmail_failure_does_not_break_the_run(parsed_factory, armed):
    """trash_message returns False rather than raising, and the node reports
    it. An untidy inbox beats a dead-lettered message that was handled fine."""
    gmail = FakeGmail(succeed=False)
    out = make_inbox_cleanup_node(gmail)(_state(parsed_factory()))
    assert "left in inbox" in out["events"][0].outcome


# --- Ingest routing ---------------------------------------------------------


def test_ingest_short_circuits_a_bounce_before_classify(parsed_factory, monkeypatch):
    """The token saving, asserted as behaviour rather than as a comment.

    A bounce must never reach the classifier — 30 of them in the observed
    mailbox were each paying for a call to be told they were not recruitment.
    """
    import app.pipeline.ingest_node as ingest_node

    monkeypatch.setattr(ingest_node, "session_scope", _no_row_session)
    out = ingest_node.ingest_node(
        {"parsed": parsed_factory(from_email="mailer-daemon@googlemail.com")}
    )
    assert out["final_status"] == MessageStatus.SKIPPED_UNDELIVERABLE


def test_ingest_leaves_ordinary_mail_alone(parsed_factory, monkeypatch):
    import app.pipeline.ingest_node as ingest_node

    monkeypatch.setattr(ingest_node, "session_scope", _no_row_session)
    out = ingest_node.ingest_node({"parsed": parsed_factory()})
    assert "final_status" not in out


class _FakeSession:
    def get(self, *_a, **_kw):
        return None


def _no_row_session():
    """Stand-in for session_scope with no matching message row."""
    from contextlib import contextmanager

    @contextmanager
    def cm():
        yield _FakeSession()

    return cm()
