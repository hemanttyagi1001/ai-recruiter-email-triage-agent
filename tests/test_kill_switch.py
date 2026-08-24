"""
Kill switch tests.

  1. is_send_halted returns the DB value (True after set_halt(value=True),
     False after set_halt(value=False)).
  2. Fail-safe: if the row is missing, is_send_halted returns True.
  3. The auto_send node no-ops on halt and sets state.halted=True.
  4. The act node no-ops on halt and sets state.halted=True.

Tests 1-2 hit the DB via the db_session fixture.
Tests 3-4 monkeypatch is_send_halted to avoid the DB path — we're
testing the node's REACTION to the flag, not the flag storage.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy import delete

from app.db.models import SystemFlag
from app.kill_switch import HALT_KEY, is_send_halted, set_halt


# ---------------------------------------------------------------------------
# DB-backed tests
# ---------------------------------------------------------------------------


def test_set_halt_true_then_read(db_session):
    """The DB round-trip round-trips."""
    # Seed the row (migration would normally do this; test DB uses
    # metadata.create_all which doesn't run migrations).
    db_session.merge(SystemFlag(key=HALT_KEY, value=False))
    db_session.flush()
    assert is_send_halted(session=db_session) is False

    row = db_session.query(SystemFlag).filter_by(key=HALT_KEY).one()
    row.value = True
    db_session.flush()
    assert is_send_halted(session=db_session) is True


def test_missing_row_fail_safe(db_session):
    """If someone dropped the seed row, is_send_halted returns True.

    That's the fail-safe choice — we don't know operator intent, so
    the safer read is 'sends halted.' See the module docstring."""
    db_session.execute(delete(SystemFlag).where(SystemFlag.key == HALT_KEY))
    db_session.flush()
    assert is_send_halted(session=db_session) is True


# ---------------------------------------------------------------------------
# Node-level tests — auto_send + act react correctly to halt
# ---------------------------------------------------------------------------


def test_auto_send_halted_returns_halted_true(parsed_factory, monkeypatch):
    """When halted, auto_send must NOT call gmail.send_reply and must
    set state.halted=True so downstream routing knows to fall back."""
    from app.pipeline import auto_send_node as asn

    monkeypatch.setattr(asn, "is_send_halted", lambda: True)

    fake_gmail = MagicMock()
    node = asn.make_auto_send_node(fake_gmail)

    parsed = parsed_factory()
    state = {"parsed": parsed, "draft_body": "hi"}
    result = node(state)

    assert result["halted"] is True
    assert "gmail_sent_id" not in result
    fake_gmail.send_reply.assert_not_called()


def test_auto_send_unhalted_calls_send_and_returns_id(parsed_factory, monkeypatch):
    from app.pipeline import auto_send_node as asn

    monkeypatch.setattr(asn, "is_send_halted", lambda: False)
    # WHY the mode has to be armed explicitly: conftest defaults the suite to
    # AUTO_SEND_MODE=off, and the node only sends on an exact match with "on".
    # Both conditions are required for a real send — an unhalted kill switch
    # alone is not sufficient — and this is the one test that proves the
    # sending path actually works, so it states both preconditions out loud.
    monkeypatch.setattr(asn.settings, "auto_send_mode", "on")

    fake_gmail = MagicMock()
    fake_gmail.send_reply.return_value = "GMAIL_MSG_123"
    node = asn.make_auto_send_node(fake_gmail)

    parsed = parsed_factory()
    state = {"parsed": parsed, "draft_body": "no thanks", "rule_verdict": None}
    result = node(state)

    assert result["halted"] is False
    assert result["gmail_sent_id"] == "GMAIL_MSG_123"
    fake_gmail.send_reply.assert_called_once()


def test_act_halted_does_not_create_draft(parsed_factory, monkeypatch):
    """The human-approved path also consults the kill switch. On halt,
    it must NOT call gmail.create_draft."""
    from app.pipeline import act as act_mod

    monkeypatch.setattr(act_mod, "is_send_halted", lambda: True)

    fake_gmail = MagicMock()
    node = act_mod.make_act_node(fake_gmail)

    parsed = parsed_factory()
    state = {
        "parsed": parsed,
        "draft_body": "hi",
        "approval_status": "approved",
    }
    result = node(state)

    assert result.get("halted") is True
    assert "gmail_draft_id" not in result
    fake_gmail.create_draft.assert_not_called()


# ---------------------------------------------------------------------------
# AUTO_SEND_MODE gating (D45)
# ---------------------------------------------------------------------------
# These sit alongside the kill-switch tests because they guard the same thing
# from a different direction: the kill switch is the runtime stop, the mode is
# the configuration stop. A real send needs BOTH to permit it.


@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_auto_send_does_not_send_unless_mode_is_on(mode, parsed_factory, monkeypatch):
    """Only the exact string "on" may send.

    GOTCHA this pins: the first version of this check asked
    `if mode == "dry_run": don't send`, which let mode="off" fall THROUGH to a
    real Gmail send — the precise opposite of what "off" means. Parametrising
    over both non-sending modes is what makes that class of inversion fail.
    """
    from app.pipeline import auto_send_node as asn

    monkeypatch.setattr(asn, "is_send_halted", lambda: False)  # switch is OPEN
    monkeypatch.setattr(asn.settings, "auto_send_mode", mode)

    fake_gmail = MagicMock()
    node = asn.make_auto_send_node(fake_gmail)

    result = node({
        "parsed": parsed_factory(),
        "draft_body": "no thanks",
        "rule_verdict": None,
    })

    fake_gmail.send_reply.assert_not_called()
    # halted=True reuses the existing "did not send" edge, so persist_auto
    # leaves the message awaiting_approval rather than marking it sent.
    assert result["halted"] is True
    assert "gmail_sent_id" not in result
    assert mode in result["events"][0].outcome


def test_kill_switch_beats_auto_send_mode_on(parsed_factory, monkeypatch):
    """AUTO_SEND_MODE=on must not override a halted kill switch.

    The two controls are independent on purpose: the mode is set at startup and
    needs a restart to change, while the kill switch is read from the database
    before every single send. If arming the mode could defeat the switch, the
    switch would stop being an emergency stop.
    """
    from app.pipeline import auto_send_node as asn

    monkeypatch.setattr(asn, "is_send_halted", lambda: True)  # switch CLOSED
    monkeypatch.setattr(asn.settings, "auto_send_mode", "on")  # fully armed

    fake_gmail = MagicMock()
    node = asn.make_auto_send_node(fake_gmail)

    result = node({
        "parsed": parsed_factory(),
        "draft_body": "no thanks",
        "rule_verdict": None,
    })

    fake_gmail.send_reply.assert_not_called()
    assert result["halted"] is True


# ---------------------------------------------------------------------------
# draft mode (D49)
# ---------------------------------------------------------------------------


def test_draft_mode_creates_draft_and_never_sends(parsed_factory, monkeypatch):
    from app.pipeline import auto_send_node as asn

    monkeypatch.setattr(asn, "is_send_halted", lambda: False)
    monkeypatch.setattr(asn.settings, "auto_send_mode", "draft")

    fake_gmail = MagicMock()
    fake_gmail.create_draft.return_value = "DRAFT_1"
    node = asn.make_auto_send_node(fake_gmail)

    result = node({
        "parsed": parsed_factory(),
        "draft_body": "hello",
        "reply_to": "hr@realcompany.com",
    })

    fake_gmail.create_draft.assert_called_once()
    fake_gmail.send_reply.assert_not_called()
    assert result["gmail_draft_id"] == "DRAFT_1"
    assert result["halted"] is False
    # The draft must go to the RESOLVED address, not the raw sender.
    assert fake_gmail.create_draft.call_args.kwargs["to"] == "hr@realcompany.com"


def test_draft_mode_ignores_kill_switch(parsed_factory, monkeypatch):
    """Drafting is permitted while halted — a draft is not outbound mail.

    This is the one place auto_send deliberately diverges from act, which does
    consult the switch before its own create_draft. See D49 for why. If this
    test ever flips, an operator who armed the switch for the soak period
    would find the agent silently doing nothing.
    """
    from app.pipeline import auto_send_node as asn

    monkeypatch.setattr(asn, "is_send_halted", lambda: True)  # switch CLOSED
    monkeypatch.setattr(asn.settings, "auto_send_mode", "draft")

    fake_gmail = MagicMock()
    fake_gmail.create_draft.return_value = "DRAFT_2"
    node = asn.make_auto_send_node(fake_gmail)

    result = node({
        "parsed": parsed_factory(),
        "draft_body": "hello",
        "reply_to": "hr@realcompany.com",
    })

    fake_gmail.create_draft.assert_called_once()
    fake_gmail.send_reply.assert_not_called()
    assert result["gmail_draft_id"] == "DRAFT_2"
    assert result["halted"] is False


def test_draft_mode_cannot_reach_the_send_call(parsed_factory, monkeypatch):
    """No state of the kill switch makes `draft` dispatch mail.

    That unreachability is what justifies drafting while halted at all — the
    mode has no path to send_reply, so leaving the switch closed costs nothing
    and opening it grants nothing.
    """
    from app.pipeline import auto_send_node as asn

    monkeypatch.setattr(asn.settings, "auto_send_mode", "draft")
    for halted in (True, False):
        monkeypatch.setattr(asn, "is_send_halted", lambda h=halted: h)
        fake_gmail = MagicMock()
        node = asn.make_auto_send_node(fake_gmail)
        node({"parsed": parsed_factory(), "draft_body": "x", "reply_to": "hr@x.com"})
        fake_gmail.send_reply.assert_not_called()
