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
    FIELD_LABELS,
    MIN_FIELDS,
    detect_fields,
    is_profile_questionnaire,
)


def _answer_labels(body: str) -> list[str]:
    """The canonical labels a rendered body actually answered, in order.

    WHY match against FIELD_LABELS rather than "every line with a colon": the
    template's own lead-in ends in one ("Here are the details you asked for:"),
    and so does the signature's confidentiality note. Keying off the real
    vocabulary is the only filter that cannot drift into counting prose.
    """
    labels = []
    for line in body.splitlines():
        head, _, rest = line.partition(":")
        if rest and head.strip() in FIELD_LABELS:
            labels.append(head.strip())
    return labels


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
    labels = _answer_labels(body)
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


# --- D68: one answer table, one format, no duplicates ------------------------


# A real pitch-plus-form, anonymised: a role description AND a screening form
# in one message. This shape classifies as a new role pitch, so it takes the
# extract/score path and never reaches build_questionnaire — which is how the
# facts came to be stated twice.
PITCH_WITH_FORM = """Hi Hemant,

We have an opening for a Senior AI Engineer with a product company in
Bangalore. Budget is 34-38 LPA. Do you have hands-on Azure AI Foundry
experience?

Kindly confirm the below details:

Current Location:
Preferred Location:
Current CTC:
Expected CTC:
Notice Period:
Current Company/Payroll:
Available for Interview:
Interested in Remote:
"""


def test_every_detectable_label_has_an_answer_behind_it():
    """The two vocabularies must stay in step.

    GOTCHA being pinned: a pattern added to _FIELD_PATTERNS with no matching
    entry in the answer table does not raise — the label is simply detected,
    counted toward MIN_FIELDS, and then silently dropped at render time. The
    recruiter gets a form back with a hole in it and nothing logs.
    """
    from app.drafts.generator import _answer_table

    assert set(_answer_table(load_profile())) == set(FIELD_LABELS)


def test_default_fact_labels_are_all_real_labels():
    from app.drafts.generator import DEFAULT_FACT_LABELS

    assert set(DEFAULT_FACT_LABELS) <= set(FIELD_LABELS)


def test_the_new_form_labels_are_detected():
    """Current Company / Available for Interview / Interested in Remote."""
    found = detect_fields(PITCH_WITH_FORM)
    assert "Current Company" in found
    assert "Available for Interview" in found
    assert "Interested in Remote" in found


def test_current_company_does_not_fire_on_a_bare_company_line():
    """A JD saying "Company: Acme" must not count as a form field.

    GOTCHA: this is the reason the pattern demands the word "current" (or
    "payroll"). Every second recruiter mail contains a bare "Company:" line
    in the role description, and matching it would hand every pitch a free
    field toward MIN_FIELDS.
    """
    assert detect_fields("Company: Acme Corp\nLocation: Pune") == []


def test_colon_is_still_required_after_an_alternation():
    """The `|` precedence bug: "a|b" + ":" means a OR b:, not (a|b):.

    Without the wrapping group, prose merely mentioning a payroll company
    would match the first alternative with no colon at all.
    """
    assert detect_fields("They asked about my current employer over lunch") == []


def test_answer_block_puts_one_field_on_each_line(parsed_factory):
    """No "·"-joined pairs. These get pasted into an ATS one field at a time.

    Also pins the no-padding rule: the HTML MIME part is a proportional font,
    so space-aligned columns would render ragged in the part most recipients
    actually see.
    """
    from app.drafts.generator import render_facts_block

    block = render_facts_block(load_profile())
    assert "·" not in block
    for line in block.splitlines():
        assert line.count(":") >= 1
        assert not line.startswith(" ")
        # One label per line means exactly one "Label: " at the front.
        assert line.split(":")[0].strip() in FIELD_LABELS


def test_a_form_on_the_llm_path_is_answered_once_in_the_senders_order(parsed_factory):
    """The D68 bug itself: the same facts rendered twice, two ways.

    The LLM path used to append a FIXED block regardless of what was asked,
    while the model separately answered the form in prose. Passing the
    detected labels into wrap_body makes the block the single answer.
    """
    from app.drafts.generator import wrap_body

    profile = load_profile()
    fields = detect_fields(PITCH_WITH_FORM)
    out = wrap_body(
        "Yes — I've worked hands-on with Azure AI Foundry.",
        parsed_factory(), None, profile, fields,
    )
    labels = _answer_labels(out)
    # Their order, not ours, and each exactly once.
    assert labels == fields
    assert len(labels) == len(set(labels))
    # The fixed default block would have led with Total Experience, which
    # this form never asked for.
    assert "Total Experience" not in labels


def test_llm_prompt_names_the_fields_the_block_will_cover():
    """Rule 9 vs FORMAT was a genuine contradiction on a form email.

    The model is no longer asked to judge what counts as a standing fact —
    it is handed the exact list the appended block will contain.
    """
    from app.drafts.llm_generator import _user_prompt

    prompt = _user_prompt(
        _P(), None, False, ["Current CTC", "Expected CTC", "Notice Period"]
    )
    assert "Current CTC" in prompt
    assert "Expected CTC" in prompt
    assert "ANSWER BLOCK IS APPENDED" in prompt


class _P:
    """Minimal stand-in for ParsedMessage — _user_prompt reads two fields."""

    subject = "Senior AI Engineer"
    body_text = PITCH_WITH_FORM


def test_unset_new_fields_are_dropped_not_guessed(parsed_factory):
    profile = load_profile()
    stripped = profile.model_copy(
        update={
            "candidate": profile.candidate.model_copy(
                update={"current_company": None}
            )
        }
    )
    body = build_questionnaire(
        parsed_factory(), stripped,
        ["Current Company", "Current CTC", "Notice Period"],
    )
    assert "Current Company" not in body
    assert "NA" not in body
