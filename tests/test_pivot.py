"""
Tests for D64: the off-field pivot and the Naukri attachment rule.

The load-bearing test in this file is
`test_solicitation_rules_can_never_produce_a_pivot`. The pivot is the only
reply shape that attaches a CV without being asked, and two of the decline
rules exist to catch paid-placement and resume-service solicitations. If a
future refactor lets a rule-fired verdict reach the score node, those scams
start receiving a CV — so that property is asserted directly rather than left
to the reader to infer from the graph's edges.
"""

from __future__ import annotations

import pytest

from app.candidate import load_profile
from app.db.models import DraftType
from app.drafts.generator import build_pivot
from app.llm.schemas import Opportunity
from app.rules.engine import build_rules, run_rules
from app.rules.resume_request import is_resume_requested, should_attach_resume


# --- Naukri attachment (bug 1) ----------------------------------------------


@pytest.mark.parametrize(
    "sender",
    [
        "priya.sharmaYnJpZ2h0cGF0aHRlY2guaW4=@naukri.com",
        "sunita.rao02bm9ydGh3aW5kLmNvbQ==@naukri.com",
        "recruiter@match.naukri.com",
    ],
)
def test_naukri_relay_always_attaches_even_with_no_request(sender):
    body = "Great opportunity for Staff Data Scientist in Hyderabad."
    # Precondition: the body genuinely contains no request, so the only reason
    # to attach is the sender.
    assert is_resume_requested(body) is False
    assert should_attach_resume(body, sender) is True


def test_non_naukri_sender_still_requires_an_actual_request():
    body = "Great opportunity for Staff Data Scientist in Hyderabad."
    assert should_attach_resume(body, "ritika@brightpathtech.in") is False


def test_non_naukri_sender_attaches_when_asked():
    body = "Kindly share your updated resume to ritika@brightpathtech.in"
    assert should_attach_resume(body, "ritika@brightpathtech.in") is True


def test_lookalike_domain_is_not_treated_as_naukri():
    """`naukrisolutions.io` is a company, not the portal."""
    body = "Staff Data Scientist role."
    assert should_attach_resume(body, "hr@naukrisolutions.io") is False


@pytest.mark.parametrize("sender", [None, "", "not-an-email"])
def test_malformed_senders_do_not_attach(sender):
    assert should_attach_resume("A role.", sender) is False


# --- The safety property (bug 3) --------------------------------------------


def _opp(**kw) -> Opportunity:
    base = dict(
        company="Acme", role_title="Staff Data Scientist", location="Bangalore",
        employment_type="permanent", ctc_min_lpa=None, ctc_max_lpa=None,
    )
    base.update(kw)
    return Opportunity(**{k: v for k, v in base.items()})


def test_any_rule_fire_routes_to_draft_and_never_to_the_scorer():
    """The structural guarantee the pivot's safety rests on.

    PIVOT is set in exactly one place — the score node. This asserts the
    scorer is unreachable whenever a rule fired, which is what keeps a
    paid-placement or resume-service solicitation from ever reaching the only
    branch that attaches a CV unasked. Asserted on the router rather than on
    any particular rule's text, because it is the EDGE that provides the
    property; a rule matching or not matching some sample string does not.
    """
    from app.pipeline.graph import _route_after_rules
    from app.rules.engine_types import RuleVerdict

    fired = RuleVerdict(
        rule_name="resume_service", verdict="decline", reason="solicitation"
    )
    assert _route_after_rules({"rule_verdict": fired}) == "draft"
    assert _route_after_rules({"rule_verdict": None}) == "score"


def test_the_solicitation_rules_are_actually_in_the_rule_list():
    """The routing guarantee above is only worth anything if these are rules.

    If paid_placement / resume_service were ever demoted out of build_rules
    and left to the fit scorer, the test above would still pass while
    solicitations started receiving a CV.
    """
    profile = load_profile()
    names = {r.__name__ for r in build_rules(profile) if hasattr(r, "__name__")}
    assert "paid_placement" in names
    assert "resume_service" in names


def test_c2h_declines_when_the_toggle_is_off():
    """The rule is still there and still works — it is opt-out, not deleted."""
    profile = load_profile().model_copy(
        update={"rules": load_profile().rules.model_copy(update={"allow_c2h": False})}
    )
    verdict = run_rules(_opp(employment_type="c2h"), build_rules(profile))
    assert verdict is not None
    assert verdict.rule_name == "c2h"


def test_c2h_reaches_the_scorer_when_the_toggle_is_on():
    """D65: with allow_c2h, a C2H role is judged on merit like any other.

    GOTCHA this pins: C2H then reaches the score node, which is the only
    place PIVOT is set — so an off-field C2H role now gets a CV attached.
    That is intended, but it is a consequence of the toggle rather than
    something the toggle says, so it is asserted rather than assumed.
    """
    profile = load_profile().model_copy(
        update={"rules": load_profile().rules.model_copy(update={"allow_c2h": True})}
    )
    assert run_rules(_opp(employment_type="c2h"), build_rules(profile)) is None


def test_the_ctc_floor_still_applies_to_c2h_when_allowed():
    """Turning C2H on does not turn off the money rule."""
    profile = load_profile().model_copy(
        update={"rules": load_profile().rules.model_copy(update={"allow_c2h": True})}
    )
    cheap_c2h = _opp(employment_type="c2h", ctc_max_lpa=12)
    verdict = run_rules(cheap_c2h, build_rules(profile))
    assert verdict is not None
    assert verdict.rule_name != "c2h"


def test_pivot_is_a_distinct_draft_type_that_fits_the_column():
    assert DraftType.PIVOT == "pivot"
    assert DraftType.PIVOT not in (DraftType.DECLINE, DraftType.INTERESTED)
    # drafts.draft_type is String(16); a longer value would truncate or error
    # depending on the driver, and neither is a thing to discover in prod.
    assert len(DraftType.PIVOT.value) <= 16


# --- Pivot body (bugs 2 and 3) ----------------------------------------------


def test_pivot_body_states_the_ai_ml_focus_and_the_attachment(parsed_factory):
    profile = load_profile()
    body = build_pivot(parsed_factory(from_name="Priya"), _opp(), profile)
    assert "AI/ML" in body
    # The template asserts an enclosure; the score node forces the flag that
    # makes it true. If this wording changes, that pairing must be rechecked.
    assert "attached my CV" in body
    assert "isn't the direction I'm heading" in body


def test_pivot_body_carries_the_new_stack_string(parsed_factory):
    """Bug 2: the experience line must describe AI/ML, not .NET / Python."""
    profile = load_profile()
    body = build_pivot(parsed_factory(), _opp(), profile)
    assert ".NET / Python" not in body
    assert "Relevant experience:" in body
    assert "GenAI" in body


def test_pivot_body_is_signed(parsed_factory):
    profile = load_profile()
    body = build_pivot(parsed_factory(), _opp(), profile)
    assert profile.drafts.signature.strip("\n") in body


def test_interested_template_no_longer_nests_parentheses(parsed_factory):
    """Bug 2: the long stack string must not render as `years (N years in (...))`."""
    from app.drafts.generator import build_interested

    profile = load_profile()
    body = build_interested(parsed_factory(), _opp(), profile)
    assert "Total experience:" in body
    assert "Relevant experience:" in body
    # The old single-bullet form wrapped stack in parens; with a stack string
    # that itself contains parens that produced unbalanced-looking output.
    assert "years (6 years in" not in body


# --- D66: uncertain no longer bypasses the threshold -------------------------


def _score_node_decision(score: int, uncertain: bool):
    """Run the real score node against a stubbed scorer and return its update.

    WHY drive the node rather than re-implement its `if`: the bug being pinned
    was a routing branch, and a test that restates the branch cannot catch the
    branch being wrong.
    """
    from unittest.mock import patch

    from app.pipeline.graph import _make_score_node
    from app.llm.client import Usage
    from app.scoring.fit import FitScore

    profile = load_profile()
    result = FitScore(score=score, rationale="stub", uncertain=uncertain)
    usage = Usage(prompt_tokens=1, completion_tokens=1, model="fake")
    with patch("app.pipeline.graph.fit_score", return_value=(result, usage)):
        node = _make_score_node(llm=None, profile=profile)
        return node({"opportunity": _opp()})


def test_uncertain_below_threshold_now_pivots_not_interested():
    """The regression that sent 'I'm interested' to a QA lead and a .NET role."""
    out = _score_node_decision(score=0, uncertain=True)
    assert out["draft_type"] == DraftType.PIVOT
    assert out["resume_requested"] is True


def test_uncertain_above_threshold_is_still_interested():
    """Abstention is not a veto — a high uncertain score still reads as a fit."""
    out = _score_node_decision(score=90, uncertain=True)
    assert out["draft_type"] == DraftType.INTERESTED


def test_confident_low_score_pivots():
    out = _score_node_decision(score=20, uncertain=False)
    assert out["draft_type"] == DraftType.PIVOT


# --- D66: the facts block is code-rendered -----------------------------------


def test_experience_renders_as_two_separate_lines():
    """Bug B: the LLM merged these into one line with a semicolon."""
    from app.drafts.generator import render_facts_block

    block = render_facts_block(load_profile())
    lines = [ln.strip() for ln in block.splitlines()]
    total = [ln for ln in lines if ln.startswith("Total Experience:")]
    relevant = [ln for ln in lines if ln.startswith("Relevant Experience:")]
    assert len(total) == 1 and len(relevant) == 1
    # The two must be on DIFFERENT lines — that is the whole bug.
    assert block.index("Total Experience") < block.index("Relevant Experience")
    assert "years; relevant" not in block.lower()


def test_wrap_body_appends_the_facts_after_the_prose(parsed_factory):
    from app.drafts.generator import wrap_body

    profile = load_profile()
    out = wrap_body("Yes, I'm available to talk Thursday.", parsed_factory(), _opp(), profile)
    assert out.index("available to talk") < out.index("Total Experience")
    assert out.index("Total Experience") < out.index(profile.drafts.signature.strip("\n"))


def test_facts_block_survives_a_missing_profile():
    from app.drafts.generator import render_facts_block

    assert render_facts_block(None) == ""
