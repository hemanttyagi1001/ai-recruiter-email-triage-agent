"""
Dead-letter writer tests.

  1. record() writes a row with the expected columns.
  2. record() swallows DB errors — a dead-letter writer that crashes
     during write is worse than one that logs and continues.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app import dead_letter
from app.db.models import DeadLetter, Message, MessageStatus, Run, RunStatus
from app.retry import PermanentExternalError


def _make_pex(exc: Exception, attempts: int = 3) -> PermanentExternalError:
    return PermanentExternalError(
        f"unit: exhausted; last was {type(exc).__name__}",
        original=exc, attempts=attempts, elapsed_ms=1234,
    )


def test_record_writes_row(db_session, monkeypatch):
    """A PermanentExternalError becomes a DeadLetter row with all fields."""
    from app import dead_letter as dl_mod

    class _Ctx:
        def __enter__(self_inner):
            return db_session
        def __exit__(self_inner, exc_type, *exc):
            if exc_type is None:
                db_session.flush()
            return False

    monkeypatch.setattr(dl_mod, "session_scope", lambda: _Ctx())

    # WHY these two rows have to exist first: dead_letters.message_id carries
    # a FK to messages.message_id, and messages.run_id a FK to runs.id. A
    # dead-letter naming a message that was never persisted violates the
    # first constraint — the INSERT fails, record() swallows the
    # IntegrityError by design (see its GOTCHA), and the test sees only a
    # bare `None` with the real cause buried in a log line.
    # This mirrors the real shape of the data: a non-null message_id can only
    # come from a failure that happened AFTER the message was persisted.
    run = Run(status=RunStatus.RUNNING)
    db_session.add(run)
    db_session.flush()
    db_session.add(
        Message(
            message_id="<test-1@x>",
            gmail_id="g-dead-letter-1",
            thread_id="thr-dead-letter-1",
            from_email="recruiter@example.com",
            from_name=None,
            subject="s",
            received_at=datetime.now(timezone.utc),
            body_text="b",
            raw_headers={},
            run_id=run.id,
            status=MessageStatus.FETCHED,
        )
    )
    db_session.flush()

    class _FakeApiError(Exception):
        pass
    _FakeApiError.__module__ = "openai"
    original = _FakeApiError("401 Unauthorized")

    row_id = dead_letter.record(
        node="classify",
        error=_make_pex(original, attempts=1),
        message_id="<test-1@x>",
        run_id=None,
    )
    assert row_id is not None

    row = db_session.query(DeadLetter).filter_by(id=row_id).one()
    assert row.node == "classify"
    assert row.message_id == "<test-1@x>"
    assert "openai" in row.error_class
    assert "401" in row.error_message
    assert row.error_details["attempts"] == 1


def test_record_swallows_db_failure_returns_none():
    """A DB blip while writing the dead-letter must not crash the caller.

    The dead-letter writer is called from the ingest loop's except-clause;
    if it raises, we lose the whole run to an error while trying to
    record a smaller error. Log-and-continue is the correct policy."""
    with patch("app.dead_letter.session_scope") as mock_scope:
        mock_scope.side_effect = RuntimeError("db down")
        pex = _make_pex(Exception("api down"))
        result = dead_letter.record(node="x", error=pex)
        assert result is None  # logged, not raised
