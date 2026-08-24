"""
Re-ingest safety. The load-bearing invariant of Phase 0 is that running
ingest twice against the same mailbox produces zero duplicate rows. Two
ways it can be broken:

  (a) Someone deletes the message_id PRIMARY KEY. DB-level test below
      guarantees the constraint stays.
  (b) The persist_node stops checking for existence before insert. Node-
      level test below covers that path.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import Message, MessageStatus, Run, RunStatus
from app.pipeline.persist import persist_node


def test_message_id_primary_key_rejects_duplicates(committed_db):
    # WHY committed_db and not db_session: this test needs real commits to
    # make the PK constraint fire at the right moment. Committing inside the
    # rollback fixture ends the transaction that fixture is holding, so its
    # teardown rollback becomes a no-op (SAWarning) and the rows it wrote
    # outlive the test. committed_db commits honestly and truncates after.
    db_session = committed_db
    run = Run(status=RunStatus.RUNNING)
    db_session.add(run)
    db_session.flush()

    def make(gmail_id: str, message_id: str) -> Message:
        return Message(
            message_id=message_id,
            gmail_id=gmail_id,
            thread_id="t",
            from_email="a@b.c",
            from_name=None,
            subject="s",
            received_at=datetime.now(timezone.utc),
            body_text="b",
            raw_headers={},
            run_id=run.id,
            status=MessageStatus.FETCHED,
        )

    db_session.add(make("g1", "<same@x>"))
    db_session.commit()

    db_session.add(make("g2", "<same@x>"))  # same PK
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_persist_node_skips_on_existing_message_id(db_session, parsed_factory, monkeypatch):
    """A second persist_node call with the same message_id must no-op, not raise."""
    # persist_node uses session_scope internally; monkeypatch it to yield our
    # transactional test session so both calls hit the same underlying transaction.
    from app.pipeline import persist as persist_mod

    class _Ctx:
        def __enter__(self_inner):
            return db_session
        def __exit__(self_inner, exc_type, *exc):
            # Session in the fixture is autoflush=False, so subsequent queries
            # won't see added rows unless we flush. Mimic session_scope's
            # commit-on-success flush semantics without actually committing —
            # the outer test transaction still rolls back for isolation.
            if exc_type is None:
                db_session.flush()
            return False  # never swallow

    monkeypatch.setattr(persist_mod, "session_scope", lambda: _Ctx())

    run = Run(status=RunStatus.RUNNING)
    db_session.add(run)
    db_session.flush()

    parsed = parsed_factory()
    state = {"parsed": parsed, "run_id": run.id, "category": "not_recruitment"}

    result_1 = persist_node(state)
    assert result_1["final_status"] == MessageStatus.SKIPPED_WRONG_CATEGORY
    assert db_session.query(Message).count() == 1

    result_2 = persist_node(state)
    # GOTCHA: this used to assert `result_2["skipped_duplicate"] is True`.
    # No code has set that key since Phase 2 split persist into
    # persist_terminal / persist_pending — grep the app package and the only
    # hit was this assertion. It failed as `None is True`, which reads like
    # a broken duplicate check rather than a stale test.
    # The behaviour under test is unchanged and still worth pinning: the
    # second call must detect the existing row (persist_terminal returns
    # early at the `s.get(Message, ...)` guard), must NOT raise, and must
    # NOT insert. Assert those directly rather than through a key that
    # described an older return shape.
    assert result_2["final_status"] == MessageStatus.SKIPPED_WRONG_CATEGORY
    assert db_session.query(Message).count() == 1  # unchanged — no second insert
