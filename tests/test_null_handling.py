"""
Null handling for absent fields.

When the email does not state a field, the extractor's Pydantic model
returns None; the persist node writes NULL to the DB. This test asserts
that path end-to-end at the ORM boundary — no silent coercion to "" or 0.
"""

from __future__ import annotations

from app.db.models import Message, MessageStatus, Opportunity as OpportunityRow, Run, RunStatus
from app.llm.schemas import Opportunity
from app.pipeline.persist import persist_node


def test_opportunity_pydantic_preserves_nulls():
    """The Pydantic layer round-trips None cleanly."""
    opp = Opportunity(
        company="Acme",
        role_title="Engineer",
        # Everything else omitted → defaults to None.
    )
    dumped = opp.model_dump()
    assert dumped["location"] is None
    assert dumped["ctc_min_lpa"] is None
    assert dumped["ctc_max_lpa"] is None
    assert dumped["recruiter_email"] is None
    assert dumped["jd_text"] is None


def test_persist_writes_nulls_not_empty_strings(db_session, parsed_factory, monkeypatch):
    """A null field on the Pydantic model must land as SQL NULL, not ''."""
    from app.pipeline import persist as persist_mod

    class _Ctx:
        def __enter__(self_inner):
            return db_session
        def __exit__(self_inner, exc_type, *exc):
            # See test_idempotency for why flush-on-success is needed here.
            if exc_type is None:
                db_session.flush()
            return False

    monkeypatch.setattr(persist_mod, "session_scope", lambda: _Ctx())

    run = Run(status=RunStatus.RUNNING)
    db_session.add(run)
    db_session.flush()

    parsed = parsed_factory(gmail_id="g-null", message_id="<null@x>")
    opp = Opportunity(company="Acme", role_title="Backend Engineer")

    state = {
        "parsed": parsed,
        "run_id": run.id,
        "category": "new_role_pitch",
        "opportunity": opp,
        "extraction_retries": 0,
        "extraction_error": None,
        # WHY reply_to is here even though this test is about NULL columns:
        # persist_terminal now reads it to distinguish "extracted fine" from
        # "extracted fine but nobody to answer" (D47). Any message carrying an
        # opportunity has already passed _route_after_extract, which cannot
        # let it through with reply_to unset — so omitting it here would
        # simulate a state the graph never produces, and this test would
        # assert against a status no real message in this shape can have.
        "reply_to": parsed.from_email,
    }
    result = persist_node(state)

    assert result["final_status"] == MessageStatus.EXTRACTED
    saved: OpportunityRow = (
        db_session.query(OpportunityRow).filter_by(message_id=parsed.message_id).one()
    )
    # Explicitly assert IS NULL, not falsy.
    assert saved.location is None
    assert saved.ctc_min_lpa is None
    assert saved.ctc_max_lpa is None
    assert saved.recruiter_email is None
    assert saved.jd_text is None
    # And the fields that WERE stated round-trip.
    assert saved.company == "Acme"
    assert saved.role_title == "Backend Engineer"


def test_ctc_range_validator_catches_inverted_range():
    """Semantic invariant check (max < min) fires at Pydantic validation time."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        Opportunity(ctc_min_lpa=40, ctc_max_lpa=20)
    assert "ctc_max_lpa" in str(exc.value)
