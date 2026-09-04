"""
What happens when the fit scorer says "I cannot judge this" (D80).

Why this file exists: the `uncertain` flag has now been wired three different
ways — D54 sent it to INTERESTED and auto-sent it, D66 removed the branch
entirely so it fell through the threshold comparison, and D80 sends it to
INTERESTED behind a mandatory human gate. Each rewiring was a one-line change
to a router and each changed what a real recruiter received. The signal is
clearly easy to get wrong, so its behaviour is pinned here rather than left
implicit in whichever router happens to read it this month.

The incident these tests encode: a Naukri alert for an "Artificial
Intelligence Engineer" role, remote, RAG/LLM/NLP — squarely on-profile. The
scorer abstained because the posting said "Not Disclosed" for salary and
returned score=0. Nothing read `uncertain`, so 0 was compared against the
threshold of 65, and the recruiter received a PIVOT saying the role "isn't
the direction I'm heading — I've moved fully into AI/ML". The scorer's own
rationale, in the same row, said the role "strongly aligns".

All offline. FakeLLM, no network, no database.
"""

from __future__ import annotations

import pytest

from app.candidate import CandidateProfile
from app.config import settings
from app.db.models import DraftType
from app.llm.schemas import FitScore


@pytest.fixture
def candidate_profile() -> CandidateProfile:
    """Same shape the other suites use. fit_threshold=65 matches the live
    profile, so the threshold comparisons here mirror production."""
    return CandidateProfile.model_validate(
        {
            "candidate": {
                "name": "Test", "total_years": 10, "relevant_years": 8,
                "stack": "AI/ML", "current_ctc_lpa": 30, "expected_ctc_lpa": 45,
                "notice_period": "30 days", "current_location": "Delhi NCR",
                "preferred_location": "Remote", "employment_status": "Employed",
            },
            "rules": {"ctc_floor_lpa": 38.0},
            "scoring": {"fit_threshold": 65},
            "drafts": {"max_length_chars": 3000},
        }
    )


# --- The score node's handling of the three cases ---------------------------


def _score_node(fake_llm, profile, usage_factory, result: FitScore):
    """Build the real score node with a queued FitScore, and run it."""
    from app.pipeline.graph import _make_score_node

    fake_llm.queue((result, usage_factory(150, 30)))
    node = _make_score_node(fake_llm, profile)
    # The node reads only state["opportunity"]; a bare object is enough.
    from app.llm.schemas import Opportunity

    return node({"opportunity": Opportunity(role_title="AI Engineer")})


def test_abstention_drafts_interested_and_demands_a_human(
    fake_llm, candidate_profile, usage_factory
):
    """The core of D80.

    Note the score is 0 AND uncertain is True — exactly the shape the real
    incident produced. Before D80 this returned PIVOT.
    """
    out = _score_node(
        fake_llm, candidate_profile, usage_factory,
        FitScore(score=0, rationale="No CTC stated; cannot weigh comp.",
                 uncertain=True),
    )
    assert out["draft_type"] == DraftType.INTERESTED
    assert out["requires_human"] is True


def test_abstention_never_produces_a_pivot(
    fake_llm, candidate_profile, usage_factory
):
    """The regression, stated as the thing that must not happen.

    PIVOT asserts a REASON — "this isn't the direction I'm heading". An
    abstention has established no reason at all, so it may never reach a
    template that claims one.
    """
    out = _score_node(
        fake_llm, candidate_profile, usage_factory,
        FitScore(score=0, rationale="Role strongly aligns, but no CTC given.",
                 uncertain=True),
    )
    assert out["draft_type"] != DraftType.PIVOT


@pytest.mark.parametrize("score", [0, 12, 64, 99])
def test_the_ignored_score_does_not_change_an_abstention(
    score, fake_llm, candidate_profile, usage_factory
):
    """`uncertain` wins over the number in both directions.

    A high score with uncertain=True must not sneak past the human gate, and a
    low one must not become a pivot. The model told us the number is noise;
    the router has to actually treat it as noise.
    """
    out = _score_node(
        fake_llm, candidate_profile, usage_factory,
        FitScore(score=score, rationale="Thin posting.", uncertain=True),
    )
    assert out["draft_type"] == DraftType.INTERESTED
    assert out["requires_human"] is True


# --- Assessed scores are untouched by D80 -----------------------------------


def test_a_confident_low_score_still_pivots(
    fake_llm, candidate_profile, usage_factory
):
    """PIVOT still exists and still fires — for an ACTUAL judgement.

    This is the case the pivot was designed for (D64): the scorer looked, and
    the role really is off-field.
    """
    out = _score_node(
        fake_llm, candidate_profile, usage_factory,
        FitScore(score=20, rationale="QA lead role, not AI/ML.", uncertain=False),
    )
    assert out["draft_type"] == DraftType.PIVOT
    assert out.get("requires_human") is not True


def test_a_confident_high_score_still_goes_interested_without_the_gate(
    fake_llm, candidate_profile, usage_factory
):
    out = _score_node(
        fake_llm, candidate_profile, usage_factory,
        FitScore(score=88, rationale="Strong GenAI match, remote.", uncertain=False),
    )
    assert out["draft_type"] == DraftType.INTERESTED
    assert out.get("requires_human") is not True


# --- The gate itself, in the router -----------------------------------------


@pytest.mark.parametrize("mode", ["dry_run", "draft", "on"])
def test_requires_human_beats_every_auto_send_mode(mode, monkeypatch):
    """The gate is not negotiable by configuration.

    AUTO_SEND_MODE=on is the operator saying "send without me". D80 says that
    consent does not extend to replies whose routing was decided by a score
    the model disclaimed. Same standing as the quarantine verdict (D14).
    """
    from app.pipeline.graph import _route_after_validate

    monkeypatch.setattr(settings, "auto_send_mode", mode)
    assert _route_after_validate({"requires_human": True}) == "persist_pending"


@pytest.mark.parametrize("mode", ["dry_run", "draft", "on"])
def test_without_the_flag_autonomy_is_unchanged(mode, monkeypatch):
    """D80 must not quietly disable the autonomy path it sits beside."""
    from app.pipeline.graph import _route_after_validate

    monkeypatch.setattr(settings, "auto_send_mode", mode)
    assert _route_after_validate({}) == "auto_send"


def test_off_mode_still_wins_when_the_flag_is_absent(monkeypatch):
    from app.pipeline.graph import _route_after_validate

    monkeypatch.setattr(settings, "auto_send_mode", "off")
    assert _route_after_validate({}) == "persist_pending"


# --- The prompt itself ------------------------------------------------------


def test_the_scorer_is_not_told_to_abstain_on_a_missing_salary(candidate_profile):
    """The literal root cause, pinned.

    The old prompt read: "Set this whenever a critical field (role, stack,
    CTC) is missing". Naukri publishes "Not Disclosed" routinely, so that
    instruction made an entire inbound channel abstain by construction. This
    asserts the instruction is gone and its replacement is present.
    """
    from app.scoring.fit import _build_system_prompt

    prompt = _build_system_prompt(candidate_profile)
    assert "role, stack, CTC" not in prompt
    assert "missing salary is NOT a reason to abstain" in prompt.replace("\n", " ")


def test_the_scorer_is_told_not_to_judge_compensation(candidate_profile):
    """Compensation is a deterministic rule that already ran (D11).

    A model re-weighing it duplicates code, and duplicates it worse — the rule
    compares a number, the model has to interpret an absent one.
    """
    from app.scoring.fit import _build_system_prompt

    flat = _build_system_prompt(candidate_profile).replace("\n", " ")
    assert "Do NOT score compensation" in flat
    assert "CTC alignment vs. expected (30%)" not in flat
