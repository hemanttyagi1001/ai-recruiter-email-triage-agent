"""
Fit scorer tests. Mock LLM — no real Azure calls.
"""

from __future__ import annotations

import pytest

from app.candidate import CandidateProfile
from app.llm.schemas import FitScore, Opportunity
from app.scoring.fit import fit_score


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile.model_validate(
        {
            "candidate": {
                "name": "Test", "total_years": 10, "relevant_years": 8,
                "stack": ".NET", "current_ctc_lpa": 30, "expected_ctc_lpa": 45,
                "notice_period": "30 days", "current_location": "Bangalore",
                "preferred_location": "Bangalore", "employment_status": "Employed",
            },
            "rules": {"ctc_floor_lpa": 38.0},
            "scoring": {"fit_threshold": 65},
            "drafts": {"max_length_chars": 3000},
        }
    )


def test_fit_score_returns_result(fake_llm, usage_factory, profile):
    fake_llm.queue(
        (FitScore(score=78, rationale="Solid stack fit, CTC in range", uncertain=False),
         usage_factory(120, 25))
    )
    opp = Opportunity(
        role_title="Senior .NET Engineer",
        company="Acme",
        location="Bangalore",
        ctc_max_lpa=50,
    )
    result, usage = fit_score(opp, profile, fake_llm)

    assert result.score == 78
    assert not result.uncertain
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 25


def test_fit_score_preserves_uncertain(fake_llm, usage_factory, profile):
    """Uncertain=True is a structural signal, not a low-score synonym."""
    fake_llm.queue(
        (FitScore(score=50, rationale="Role too vague to judge", uncertain=True),
         usage_factory())
    )
    opp = Opportunity(role_title="Great opportunity")  # deliberately thin
    result, _ = fit_score(opp, profile, fake_llm)

    assert result.uncertain is True
    # Downstream MUST treat this as route-to-review, ignoring the score.
    # (That routing lives in graph.py; here we just assert the field
    # survives the round-trip.)


def test_fit_score_calls_llm_with_opportunity_json(fake_llm, usage_factory, profile):
    """Sanity: the scorer feeds structured JSON, not the raw email body."""
    fake_llm.queue((FitScore(score=60, rationale="ok", uncertain=False), usage_factory()))
    opp = Opportunity(role_title="Backend Engineer", company="Widget Co", ctc_max_lpa=45)
    fit_score(opp, profile, fake_llm)

    _, messages = fake_llm.calls[-1]
    user_content = [m["content"] for m in messages if m["role"] == "user"][0]
    assert "Widget Co" in user_content
    assert "Backend Engineer" in user_content


def test_score_range_validated_by_pydantic():
    """The schema itself enforces 0..100; a bug in the model can't leak > 100."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FitScore(score=150, rationale="x", uncertain=False)
    with pytest.raises(ValidationError):
        FitScore(score=-5, rationale="x", uncertain=False)
