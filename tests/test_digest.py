"""
Digest CLI tests — seed a bit of data, assert the output reflects it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from app.cli.digest import build_digest, format_text
from app.db.models import (
    DeadLetter,
    Draft,
    DraftStatus,
    DraftType,
    DuplicateFlag,
    Message,
    MessageStatus,
    Opportunity,
    Run,
    RunStatus,
    SystemFlag,
)
from app.kill_switch import HALT_KEY


def test_digest_counts_auto_declines_and_awaiting(db_session, monkeypatch):
    """Seed one auto-decline (recent) + one awaiting_approval + one
    dead_letter + one dupe flag + one finished run — assert every
    counter reflects them."""
    from app.cli import digest as digest_mod
    from app import kill_switch as ks_mod

    class _Ctx:
        def __enter__(self_inner):
            return db_session
        def __exit__(self_inner, exc_type, *exc):
            if exc_type is None:
                db_session.flush()
            return False
    monkeypatch.setattr(digest_mod, "session_scope", lambda: _Ctx())
    monkeypatch.setattr(ks_mod, "session_scope", lambda: _Ctx())

    # Seed the halt flag row so is_send_halted has something to read.
    db_session.merge(SystemFlag(key=HALT_KEY, value=False))
    db_session.flush()

    now = datetime.now(timezone.utc)

    # Run row (finished within period).
    run = Run(
        id=uuid4(),
        started_at=now - timedelta(hours=2),
        finished_at=now - timedelta(hours=1),
        status=RunStatus.SUCCEEDED,
        classify_tokens_in=100, classify_tokens_out=20,
        extract_tokens_in=400, extract_tokens_out=150,
        score_tokens_in=200, score_tokens_out=40,
        embed_tokens_in=300,
        estimated_cost_usd=Decimal("0.0123"),
    )
    db_session.add(run)
    db_session.flush()

    # Auto-sent message + opportunity + draft.
    auto_mid = "<auto-1@x>"
    db_session.add(Message(
        message_id=auto_mid, gmail_id="g-auto-1", thread_id="thr-auto-1",
        from_email="r@x.com", from_name="R", subject="s",
        received_at=now - timedelta(hours=3),
        body_text="b", raw_headers={}, run_id=run.id,
        status=MessageStatus.AUTO_SENT,
    ))
    opp = Opportunity(id=uuid4(), message_id=auto_mid, employment_type="c2h")
    db_session.add(opp)
    db_session.flush()
    db_session.add(Draft(
        message_id=auto_mid, opportunity_id=opp.id,
        draft_type=DraftType.DECLINE, body_text="no thanks",
        status=DraftStatus.AUTO_SENT, auto_actioned=True,
        rule_name="c2h", rule_reason="C2H roles auto-decline",
        resolved_at=now - timedelta(hours=1),
    ))

    # Awaiting_approval message.
    await_mid = "<await-1@x>"
    db_session.add(Message(
        message_id=await_mid, gmail_id="g-await-1", thread_id="thr-await-1",
        from_email="r2@x.com", from_name="R2", subject="s2",
        received_at=now - timedelta(hours=6),
        body_text="b2", raw_headers={}, run_id=run.id,
        status=MessageStatus.AWAITING_APPROVAL,
    ))

    # Duplicate flag between two synthetic opps (need a second opp).
    opp2 = Opportunity(id=uuid4(), message_id=None)
    # Actually opp requires message_id — reuse a message. Simpler:
    # create a second message + opp.
    dupe_mid = "<dupe-1@x>"
    db_session.add(Message(
        message_id=dupe_mid, gmail_id="g-dupe-1", thread_id="thr-dupe-1",
        from_email="r3@x.com", from_name="R3", subject="s3",
        received_at=now - timedelta(hours=4),
        body_text="b3", raw_headers={}, run_id=run.id,
        status=MessageStatus.EXTRACTED,
    ))
    opp2 = Opportunity(id=uuid4(), message_id=dupe_mid)
    db_session.add(opp2)
    db_session.flush()
    db_session.add(DuplicateFlag(
        opportunity_id=opp2.id, matched_opportunity_id=opp.id,
        similarity=Decimal("0.9"),
        flagged_at=now - timedelta(hours=2),
    ))

    # Dead letter.
    db_session.add(DeadLetter(
        node="classify", error_class="openai.RateLimitError",
        error_message="429 quota", error_details={"attempts": 5},
        occurred_at=now - timedelta(hours=1),
        run_id=run.id,
    ))
    db_session.flush()

    d = build_digest(hours=24)

    assert d.auto_declines_sent == 1
    assert d.auto_declines_by_rule == {"c2h": 1}
    assert d.awaiting_approval == 1
    assert d.awaiting_oldest_hours is not None
    assert d.dead_lettered == 1
    assert d.duplicate_flags_raised == 1
    assert d.runs_in_period == 1
    assert d.kill_switch_on is False
    assert d.cost_usd == "0.0123"

    text = format_text(d)
    assert "Auto-declines sent          : 1" in text
    assert "c2h=1" in text
    assert "Awaiting human approval     : 1" in text
    assert "Dead-lettered failures      : 1" in text
    assert "Kill switch                 : OFF" in text
