"""
Tests for D67: answering a screening form without thread history.

The load-bearing tests here are the NEGATIVE ones. This feature reopens a
slice of D55, and the whole safety argument is that the slice is narrow —
so the tests that matter most are the ones proving an ordinary follow-up, a
CTC negotiation, and a new role pitch all still take their old paths.

`test_giridhar_shaped_form_is_answered_in_the_senders_order` is built from
the real message that motivated this (anonymised), because a detector tuned
on invented samples proves only that it matches invented samples.
"""

from __future__ import annotations

import pytest

from app.candidate import load_profile
from app.db.models import Category, DraftType
from app.drafts.generator import build_questionnaire
from app.rules.questionnaire import (
    MIN_FIELDS,
    detect_fields,
    is_profile_questionnaire,
)


# A real screening form, anonymised. This exact shape was skipped in
# production on 2026-08-26 and is why this module exists.
FORM = """Hi Hemant,

Thank you for sharing your resume and expressing interest in the opportunity.
To proceed further, kindly share the following details at your earliest
convenience:

Current Location:
Native Location:
Total Experience:
Relevant Experience:
Preferred Location to Work:
Reason for Job Change:
Notice Period (If currently not working, please mention last working day):
Current CTC:
Expected CTC:

Looking forward to your positive response!
"""


# --- Detection ---------------------------------------------------------------


def test_the_real_form_is_detected():
    assert is_profile_questionnaire(FORM) is True


def test_fields_come_back_in_the_senders_order():
    """Order is what makes the reply paste-able into their ATS."""
    assert detect_fields(FORM) == [
        "Current Location",
        "Native Location",
        "Total Experience",
        "Relevant Experience",
        "Preferred Location",
        "Reason for Job Change",
        "Notice Period",
        "Current CTC",
        "Expected CTC",
    ]


# --- The negative cases: D55 must still hold ---------------------------------


def test_a_ctc_negotiation_is_not_a_questionnaire():
    """The single most common follow-up, and exactly what D55 was about.

    Answering this with a wall of salary figures would be worse than silence.
    """
    body = (
        "Hi Hemant, thanks for your time. The client has come back and their "
        "budget for this role tops out around 30 LPA against your expected "
        "CTC. Would you consider it?"
    )
    assert is_profile_questionnaire(body) is False


def test_a_rejection_is_not_a_questionnaire():
    body = (
        "Hi Hemant, unfortunately we do not have any openings that match your "
        "skill set at the moment. We will keep your profile on file."
    )
    assert is_profile_questionnaire(body) is False


def test_one_field_alone_is_not_a_form():
    """A single ask is a conversation; a list is paperwork."""
    assert is_profile_questionnaire("Could you confirm your Notice Period: ?") is False


def test_two_fields_are_still_below_the_threshold():
    body = "Please confirm.\nCurrent CTC:\nExpected CTC:"
    assert len(detect_fields(body)) == 2
    assert is_profile_questionnaire(body) is False


def test_three_fields_is_the_boundary():
    body = "Current CTC:\nExpected CTC:\nNotice Period:"
    assert len(detect_fields(body)) == MIN_FIELDS
    assert is_profile_questionnaire(body) is True


def test_prose_mentioning_ctc_without_a_colon_does_not_match():
    """The trailing colon is most of the discrimination — pin it."""
    body = (
        "What is your current CTC and expected CTC, and what notice period "
        "are you on right now?"
    )
    assert is_profile_questionnaire(body) is False


@pytest.mark.parametrize("body", [None, "", "   "])
def test_empty_bodies_are_safe(body):
    assert is_profile_questionnaire(body) is False
    assert detect_fields(body) == []


# --- Routing: both category AND content must agree ---------------------------


def _route(category, body):
    from app.pipeline.graph import _route_after_classify

    from tests.conftest import make_parsed

    return _route_after_classify(
        {"category": category, "parsed": make_parsed(body=body)}
    )


def test_a_followup_form_routes_to_the_questionnaire_node():
    assert _route(Category.FOLLOWUP_EXISTING_THREAD, FORM) == "questionnaire"


def test_an_ordinary_followup_still_terminates():
    """D55 unchanged for everything that is not a form."""
    assert _route(
        Category.FOLLOWUP_EXISTING_THREAD, "Any update on the below?"
    ) == "persist_terminal"


def test_a_new_role_pitch_containing_a_form_still_goes_to_extract():
    """Content alone must not divert a pitch away from scoring.

    A pitch that lists three field labels is still a pitch — it needs
    extraction, rules and a fit score, not a reflexive form-fill.
    """
    assert _route(Category.NEW_ROLE_PITCH, FORM) == "extract"


def test_not_recruitment_still_terminates_even_with_a_form():
    assert _route(Category.NOT_RECRUITMENT, FORM) == "persist_terminal"


# --- The reply body ----------------------------------------------------------


def test_giridhar_shaped_form_is_answered_in_the_senders_order(parsed_factory):
    profile = load_profile()
    body = build_questionnaire(
        parsed_factory(from_name="Giri"), profile, detect_fields(FORM)
    )
    labels = [
        ln.split(":")[0].strip(" -")
        for ln in body.splitlines()
        if ln.strip().startswith("- ")
    ]
    assert labels[:4] == [
        "Current Location", "Native Location", "Total Experience",
        "Relevant Experience",
    ]
    assert "Expected CTC" in labels


def test_unknown_fields_are_dropped_never_guessed(parsed_factory):
    """A value the profile does not hold must not appear at all.

    GOTCHA being pinned: rendering "NA" next to Expected CTC reads as evasive
    on the exact question being screened for, and inventing one puts words in
    the candidate's mouth. Dropping is the only honest option.
    """
    profile = load_profile()
    stripped = profile.model_copy(
        update={
            "candidate": profile.candidate.model_copy(
                update={"reason_for_job_change": None}
            )
        }
    )
    body = build_questionnaire(
        parsed_factory(), stripped, ["Current CTC", "Reason for Job Change", "Notice Period"]
    )
    assert "Reason for Job Change" not in body
    assert "NA" not in body
    assert "Current CTC" in body


def test_no_answerable_fields_raises_rather_than_sending_an_empty_list(parsed_factory):
    profile = load_profile()
    with pytest.raises(ValueError):
        build_questionnaire(parsed_factory(), profile, [])


def test_questionnaire_body_is_signed_and_carries_no_cv_claim(parsed_factory):
    profile = load_profile()
    body = build_questionnaire(parsed_factory(), profile, detect_fields(FORM))
    assert profile.drafts.signature.strip("\n") in body
    # A follow-up thread means they already have the CV; claiming an
    # attachment here would be false, since the node sets resume_requested
    # False.
    assert "attached" not in body.lower()


def test_questionnaire_is_a_distinct_draft_type_that_fits_the_column():
    assert DraftType.QUESTIONNAIRE == "questionnaire"
    assert len(DraftType.QUESTIONNAIRE.value) <= 16
