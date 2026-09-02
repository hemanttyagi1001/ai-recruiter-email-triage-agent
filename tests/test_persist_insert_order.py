"""
Regression tests for the INSERT ordering incident of 2026-09-02 (D78).

What broke and why a test file exists for it: `persist_auto` and
`persist_pending` each write a `messages` row and a `drafts` row in ONE
transaction. `Draft.message_id` carried a ForeignKey but the ORM had no
`relationship()` between the two mappers, and SQLAlchemy builds its flush
order from relationships — not from raw ForeignKey columns. So the unit of
work was free to emit `INSERT INTO drafts` first, Postgres rejected it on
`drafts_message_id_fkey`, and the whole transaction rolled back.

Why that was expensive rather than merely wrong: by the time `persist_auto`
runs, `auto_send` has ALREADY created the Gmail draft. A failed transaction
therefore left a real draft in the mailbox with nothing in the database
pointing at it — so the next ingest cycle's dedup guards saw an unseen
message and drafted for it again. Once every POLL_INTERVAL_MINUTES, forever.
Twelve drafts for two recruiters before it was noticed.

Why it hid for months: every path that carries an Opportunity calls
`s.flush()` to populate `opp_row.id` for a downstream FK, and that flush
forced `messages` out first BY ACCIDENT. The questionnaire path (D67) is the
only route to `persist_auto` with `opportunity is None`, so it was the first
to have no flush, and it went straight into production behaviour.

These tests assert the OUTCOME (both rows land) rather than the mechanism
(what order the INSERTs came out in). A test that pinned the SQL order would
pass just as happily if a future refactor dropped the relationship and
reintroduced the accidental-flush dependency.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect as sa_inspect

from app.db.models import Draft, Message


# --- The ORM declaration that does the ordering ------------------------------


def test_draft_and_message_are_related_in_the_orm_not_just_in_the_schema():
    """The root cause, asserted directly.

    A ForeignKey column alone is invisible to the unit of work. If this
    relationship is ever removed, INSERT ordering silently reverts to
    depending on an incidental flush elsewhere in the same function — which
    is exactly the state that produced the incident.
    """
    mapper = sa_inspect(Message)
    assert "draft" in mapper.relationships, (
        "Message.draft relationship is missing — without it SQLAlchemy has no "
        "dependency edge between messages and drafts and may INSERT the child "
        "first. See D78."
    )
    assert "message" in sa_inspect(Draft).relationships


def test_the_dependency_runs_from_message_to_draft():
    """Direction matters: messages is the parent and must be written first."""
    rel = sa_inspect(Message).relationships["draft"]
    assert rel.mapper.class_ is Draft
    assert rel.back_populates == "message"


# --- The behaviour the incident actually broke -------------------------------


@pytest.mark.parametrize("with_opportunity", [False, True])
def test_message_and_draft_commit_together_in_one_transaction(
    committed_db, parsed_factory, with_opportunity
):
    """Both rows land, with and without an Opportunity in the same flush.

    WHY parametrised on `with_opportunity`: that flag is the entire
    difference between the path that worked for months and the path that
    broke. `True` reproduces the accidental protection (an intervening
    flush); `False` reproduces the questionnaire path that had none. Only the
    second one ever failed, and testing only the second would let someone
    "fix" this by deleting the opportunity flush and still pass.
    """
    from app.db.models import (
        DraftStatus,
        DraftType,
        MessageStatus,
        Opportunity as OpportunityRow,
        Run,
        RunStatus,
    )

    run = Run(status=RunStatus.RUNNING)
    committed_db.add(run)
    committed_db.flush()
    committed_db.commit()

    parsed = parsed_factory(gmail_id="g-order-1")

    # Deliberately mirrors persist_auto's write order: Message, then
    # (optionally) Opportunity, then Draft — all in one session.
    committed_db.add(
        Message(
            message_id=parsed.message_id,
            gmail_id=parsed.gmail_id,
            thread_id=parsed.thread_id,
            from_email=parsed.from_email,
            from_name=parsed.from_name,
            subject=parsed.subject,
            received_at=parsed.received_at,
            body_text=parsed.body_text,
            raw_headers=parsed.raw_headers,
            run_id=run.id,
            category="followup_existing_thread",
            status=MessageStatus.SENT_TO_GMAIL_DRAFTS,
        )
    )

    opp_id = None
    if with_opportunity:
        opp_row = OpportunityRow(message_id=parsed.message_id, company="Acme")
        committed_db.add(opp_row)
        committed_db.flush()
        opp_id = opp_row.id

    committed_db.add(
        Draft(
            message_id=parsed.message_id,
            opportunity_id=opp_id,
            draft_type=DraftType.QUESTIONNAIRE,
            body_text="Here are the details you asked for.",
            status=DraftStatus.SENT_TO_GMAIL_DRAFTS,
            auto_actioned=True,
            reply_to_email="info@example.org",
        )
    )

    # Before D78 this raised IntegrityError (ForeignKeyViolation) whenever
    # with_opportunity was False.
    committed_db.commit()

    assert committed_db.get(Message, parsed.message_id) is not None
    assert (
        committed_db.query(Draft).filter_by(message_id=parsed.message_id).one_or_none()
        is not None
    )


def test_a_persisted_message_is_not_re_ingested(committed_db, parsed_factory):
    """The consequence that made the bug expensive, asserted end to end.

    The duplicate drafting was never a drafting bug — it was this lookup
    returning None because the write it depended on had rolled back. Pinning
    it here means a future regression shows up as "the guard stopped working"
    rather than as a mailbox filling with drafts.
    """
    from app.db.models import MessageStatus, Run, RunStatus

    run = Run(status=RunStatus.RUNNING)
    committed_db.add(run)
    committed_db.flush()
    committed_db.commit()

    parsed = parsed_factory(gmail_id="g-order-2")
    committed_db.add(
        Message(
            message_id=parsed.message_id,
            gmail_id=parsed.gmail_id,
            thread_id=parsed.thread_id,
            from_email=parsed.from_email,
            from_name=parsed.from_name,
            subject=parsed.subject,
            received_at=parsed.received_at,
            body_text=parsed.body_text,
            raw_headers=parsed.raw_headers,
            run_id=run.id,
            status=MessageStatus.SENT_TO_GMAIL_DRAFTS,
        )
    )
    committed_db.commit()

    # Both guards ingest._process_one applies before invoking the graph.
    assert (
        committed_db.query(Message.gmail_id).filter_by(gmail_id=parsed.gmail_id).first()
        is not None
    )
    assert committed_db.get(Message, parsed.message_id) is not None
