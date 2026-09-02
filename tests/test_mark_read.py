"""
Tests for D68: mark the inbound message read once a reply has actually gone.

The important test in this file is
`test_a_mark_read_failure_does_not_break_the_send`. Everything else here is
cosmetic — an unread label is a nicety. But if a mark_read failure could
propagate, the ingest CLI would dead-letter a message whose reply had ALREADY
been delivered, and a later re-ingest would send the recruiter a second copy.
That turns a cosmetic bug into the duplicate-reply failure this project has
already been burned by, so it is pinned directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.pipeline import auto_send_node as asn


def _state(parsed):
    return {"parsed": parsed, "draft_body": "hi"}


def test_marks_read_after_a_successful_send(parsed_factory, monkeypatch):
    monkeypatch.setattr(asn, "is_send_halted", lambda: False)
    monkeypatch.setattr(asn.settings, "auto_send_mode", "on")

    gmail = MagicMock()
    gmail.send_reply.return_value = "sent-1"
    gmail.mark_read.return_value = True

    parsed = parsed_factory(gmail_id="g-42")
    result = asn.make_auto_send_node(gmail)(_state(parsed))

    assert result["gmail_sent_id"] == "sent-1"
    # GOTCHA pinned: addressed by Gmail's internal id, not the RFC 5322
    # Message-ID that is our primary key. modify() rejects the latter.
    gmail.mark_read.assert_called_once_with("g-42")


def test_a_mark_read_failure_does_not_break_the_send(parsed_factory, monkeypatch):
    """The load-bearing one. See the module docstring.

    A raising mark_read must not escape the node — the email is already
    delivered, and an exception here would dead-letter a message that was
    successfully answered, setting up a duplicate on the next ingest.
    """
    monkeypatch.setattr(asn, "is_send_halted", lambda: False)
    monkeypatch.setattr(asn.settings, "auto_send_mode", "on")

    gmail = MagicMock()
    gmail.send_reply.return_value = "sent-2"
    gmail.mark_read.side_effect = RuntimeError("insufficientPermissions")

    parsed = parsed_factory(gmail_id="g-43")

    # The real client swallows this internally; belt-and-braces, the node must
    # survive even a client that does raise.
    try:
        result = asn.make_auto_send_node(gmail)(_state(parsed))
    except RuntimeError:
        pytest.fail("a mark_read failure escaped auto_send and would dead-letter "
                    "a message whose reply was already delivered")

    assert result["gmail_sent_id"] == "sent-2"
    assert result["halted"] is False


def test_halted_send_marks_nothing(parsed_factory, monkeypatch):
    """Nothing was sent, so nothing has been answered."""
    monkeypatch.setattr(asn, "is_send_halted", lambda: True)

    gmail = MagicMock()
    asn.make_auto_send_node(gmail)(_state(parsed_factory()))

    gmail.send_reply.assert_not_called()
    gmail.mark_read.assert_not_called()


@pytest.mark.parametrize("mode", ["dry_run", "draft"])
def test_non_sending_modes_mark_nothing(parsed_factory, monkeypatch, mode):
    """`read` must mean `answered`, so a draft that never left does not count."""
    monkeypatch.setattr(asn, "is_send_halted", lambda: False)
    monkeypatch.setattr(asn.settings, "auto_send_mode", mode)

    gmail = MagicMock()
    gmail.create_draft.return_value = "draft-1"
    asn.make_auto_send_node(gmail)(_state(parsed_factory()))

    gmail.mark_read.assert_not_called()


# --- The client method itself ------------------------------------------------


def test_client_mark_read_never_raises():
    """It is the one method on GmailClient that must always return."""
    from app.gmail.client import GmailClient

    service = MagicMock()
    service.users.return_value.messages.return_value.modify.return_value.execute.side_effect = (
        RuntimeError("insufficientPermissions")
    )
    assert GmailClient(service).mark_read("g-1") is False


def test_client_mark_read_removes_only_the_unread_label():
    from app.gmail.client import GmailClient

    service = MagicMock()
    assert GmailClient(service).mark_read("g-1") is True
    _, kwargs = service.users.return_value.messages.return_value.modify.call_args
    assert kwargs["id"] == "g-1"
    assert kwargs["body"] == {"removeLabelIds": ["UNREAD"]}
    # Nothing is ARCHIVED or deleted — the message stays in the inbox.
    assert "addLabelIds" not in kwargs["body"]
    assert "removeLabelIds" in kwargs["body"]
    assert "INBOX" not in kwargs["body"]["removeLabelIds"]


def test_modify_scope_is_requested():
    from app.gmail.client import SCOPES

    assert "https://www.googleapis.com/auth/gmail.modify" in SCOPES
    # D40: readonly and compose must survive alongside it.
    assert "https://www.googleapis.com/auth/gmail.readonly" in SCOPES
    assert "https://www.googleapis.com/auth/gmail.compose" in SCOPES
