"""
Rule engine tests.

Covers each rule's happy and unhappy paths, and the engine's first-match-
wins ordering. All tests are pure — no LLM, no DB — since the rules are
plain functions of Opportunity.
"""

from __future__ import annotations

import pytest

from app.candidate import CandidateProfile
from app.llm.schemas import Opportunity
from app.rules import decliners
from app.rules.engine import build_rules, run_rules


# --- Fixtures ---


@pytest.fixture
def profile() -> CandidateProfile:
    """A minimal profile stub — same shape as candidate.toml."""
    return CandidateProfile.model_validate(
        {
            "candidate": {
                "name": "Test User",
                "total_years": 10,
                "relevant_years": 8,
                "stack": ".NET",
                "current_ctc_lpa": 30,
                "expected_ctc_lpa": 45,
                "notice_period": "30 days",
                "current_location": "Bangalore",
                "preferred_location": "Bangalore or remote",
                "employment_status": "Employed",
            },
            "rules": {"ctc_floor_lpa": 38.0},
            "scoring": {"fit_threshold": 65},
            "drafts": {"max_length_chars": 3000},
        }
    )


# --- Rule 1: c2h ---


def test_c2h_declines():
    v = decliners.is_c2h(Opportunity(employment_type="c2h"))
    assert v is not None
    assert v.rule_name == "c2h"
    assert v.verdict == "decline"


def test_permanent_passes():
    assert decliners.is_c2h(Opportunity(employment_type="permanent")) is None


def test_unstated_employment_type_passes():
    assert decliners.is_c2h(Opportunity()) is None


# --- Rule 2: ctc below floor ---


def test_ctc_below_floor_declines():
    rule = decliners.make_ctc_below_floor(38.0)
    v = rule(Opportunity(ctc_max_lpa=30))
    assert v is not None
    assert "30" in v.reason and "38" in v.reason


def test_ctc_at_or_above_floor_passes():
    rule = decliners.make_ctc_below_floor(38.0)
    assert rule(Opportunity(ctc_max_lpa=38)) is None
    assert rule(Opportunity(ctc_max_lpa=50)) is None


def test_unstated_ctc_does_not_decline():
    """Silent when unstated — negotiation-friendly roles omit numbers."""
    rule = decliners.make_ctc_below_floor(38.0)
    assert rule(Opportunity(ctc_max_lpa=None)) is None


# --- Rule 3: outside India, no sponsorship ---


def test_dubai_no_sponsorship_declines():
    v = decliners.outside_india_no_sponsorship(
        Opportunity(location="Dubai, UAE", work_model="onsite", jd_text="great role")
    )
    assert v is not None


def test_dubai_with_sponsorship_passes():
    v = decliners.outside_india_no_sponsorship(
        Opportunity(
            location="Dubai, UAE",
            work_model="onsite",
            jd_text="We provide full visa sponsorship and a relocation package.",
        )
    )
    assert v is None


def test_remote_role_passes_regardless_of_location():
    """Remote is location-agnostic. HQ in Berlin doesn't matter."""
    v = decliners.outside_india_no_sponsorship(
        Opportunity(location="Berlin, Germany", work_model="remote")
    )
    assert v is None


def test_bangalore_passes():
    v = decliners.outside_india_no_sponsorship(
        Opportunity(location="Bangalore, Karnataka", work_model="hybrid")
    )
    assert v is None


def test_bengaluru_alternate_spelling_passes():
    v = decliners.outside_india_no_sponsorship(
        Opportunity(location="Bengaluru", work_model="onsite")
    )
    assert v is None


def test_ncr_passes():
    assert (
        decliners.outside_india_no_sponsorship(
            Opportunity(location="Delhi NCR", work_model="onsite")
        )
        is None
    )


def test_unstated_location_passes():
    """Can't judge → don't decline on this rule."""
    assert decliners.outside_india_no_sponsorship(Opportunity()) is None


# --- Rule 4: paid placement ---


@pytest.mark.parametrize(
    "text",
    [
        "We charge a small processing fee of Rs 5000.",
        "This role has a one-time registration fee.",
        "Candidate pays the placement fee.",
        "A refundable deposit is required.",
        "Nominal fee for training applies.",
    ],
)
def test_paid_placement_detects_variants(text):
    v = decliners.paid_placement(Opportunity(jd_text=text))
    assert v is not None
    assert v.rule_name == "paid_placement"


def test_clean_jd_passes_paid_placement():
    v = decliners.paid_placement(
        Opportunity(jd_text="Senior .NET role, 8+ years experience, Bangalore.")
    )
    assert v is None


# --- Rule 5: resume service ---


@pytest.mark.parametrize(
    "role_title",
    [
        "Resume Writing Services - Get shortlisted!",
        "LinkedIn Optimization Package",
        "CV Makeover for Senior Professionals",
        "ATS-Friendly Resume Building",
    ],
)
def test_resume_service_detects_variants(role_title):
    v = decliners.resume_service(Opportunity(role_title=role_title))
    assert v is not None


def test_clean_role_title_passes_resume_service():
    v = decliners.resume_service(Opportunity(role_title="Senior Backend Engineer"))
    assert v is None


# --- Engine ---


def test_engine_first_match_wins(profile):
    """c2h fires before ctc_below_floor when both would."""
    rules = build_rules(profile)
    opp = Opportunity(employment_type="c2h", ctc_max_lpa=20)
    v = run_rules(opp, rules)
    assert v is not None
    assert v.rule_name == "c2h"


def test_engine_returns_none_when_all_pass(profile):
    rules = build_rules(profile)
    opp = Opportunity(
        employment_type="permanent",
        ctc_max_lpa=50,
        location="Bangalore",
        work_model="hybrid",
        jd_text="Backend engineering role at a fintech.",
    )
    assert run_rules(opp, rules) is None


def test_engine_returns_none_on_empty_opportunity(profile):
    """Fully-null opportunity means the extractor found nothing — pass to scorer."""
    rules = build_rules(profile)
    assert run_rules(Opportunity(), rules) is None
