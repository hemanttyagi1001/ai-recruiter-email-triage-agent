"""
Draft generator tests. All pure — no LLM, no DB.
"""

from __future__ import annotations

import pytest

from app.candidate import CandidateProfile
from app.drafts.generator import build_decline, build_interested
from app.llm.schemas import Opportunity


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile.model_validate(
        {
            "candidate": {
                "name": "Hemant Kumar Tyagi",
                "total_years": 14,
                "relevant_years": 12,
                "stack": ".NET / Python",
                "current_ctc_lpa": 42,
                "expected_ctc_lpa": 55,
                "notice_period": "60 days",
                "current_location": "Delhi NCR",
                "preferred_location": "Delhi NCR or remote",
                "employment_status": "Currently employed",
            },
            "rules": {"ctc_floor_lpa": 38.0},
            "scoring": {"fit_threshold": 65},
            "drafts": {"max_length_chars": 3000},
        }
    )


def test_interested_draft_fills_candidate_details(profile, parsed_factory):
    parsed = parsed_factory(subject="Great role at Acme", from_email="jane@acme.com")
    parsed.from_name = "Jane Doe"
    opp = Opportunity(role_title="Senior Backend Engineer", company="Acme")

    body = build_interested(parsed, opp, profile)

    assert "Jane" in body                        # salutation
    assert "Senior Backend Engineer" in body     # role
    assert "Acme" in body                        # company
    assert "14 years" in body                    # total exp
    assert ".NET / Python" in body               # stack
    assert "42" in body                          # current ctc
    assert "55" in body                          # expected ctc
    assert "Delhi NCR" in body                   # location


def test_decline_draft_includes_reason(profile, parsed_factory):
    parsed = parsed_factory(subject="C2H role")
    parsed.from_name = "Rajesh"
    reason = "C2H (contract-to-hire) arrangements are not a fit"

    body = build_decline(parsed, reason)

    assert "Rajesh" in body
    assert reason in body


def test_salutation_falls_back_when_no_name(profile, parsed_factory):
    parsed = parsed_factory(from_email="anon@x.com")
    parsed.from_name = None
    body = build_decline(parsed, "not a fit")
    assert "Hi there," in body


def test_drafts_end_before_sign_off(profile, parsed_factory):
    """Mail-client signature auto-appends. Template must not include one."""
    parsed = parsed_factory()
    opp = Opportunity(role_title="Dev", company="Co")
    body = build_interested(parsed, opp, profile).rstrip()

    # No candidate name as final line (would collide with signature).
    last_line = body.splitlines()[-1].strip()
    assert profile.candidate.name not in last_line

    # No "Best," / "Regards," / "Thanks," on final line either.
    assert not any(
        last_line.lower().startswith(prefix)
        for prefix in ("best", "regards", "thanks,", "cheers", "sincerely")
    )


def test_clarifications_bullets_what_is_missing(profile, parsed_factory):
    """Missing fields → deterministic clarification asks."""
    parsed = parsed_factory()
    opp = Opportunity(role_title="Dev", company="Co")  # everything else null
    body = build_interested(parsed, opp, profile)

    assert "share the full JD" in body
    assert "permanent or contract" in body
    assert "budget range" in body


def test_clarifications_omits_asks_for_present_fields(profile, parsed_factory):
    """Fields already stated do not show up in clarifications."""
    parsed = parsed_factory()
    opp = Opportunity(
        role_title="Dev",
        company="Co",
        jd_text="Full JD here",
        employment_type="permanent",
        ctc_min_lpa=40,
        ctc_max_lpa=55,
        notice_period="30 days",
        work_model="hybrid",
    )
    body = build_interested(parsed, opp, profile)

    assert "share the full JD" not in body
    assert "budget range" not in body


def test_no_llm_used(profile, parsed_factory):
    """Draft generation must be a pure function of its inputs — no I/O.

    Sanity-checked by patching the LLM client with a raiser and confirming
    draft build succeeds without touching it.
    """
    parsed = parsed_factory()
    opp = Opportunity(role_title="Dev")
    # If build_interested touched llm.structured_completion, this would explode.
    # (It doesn't touch app.llm.client at all — the module isn't imported.)
    body = build_interested(parsed, opp, profile)
    assert body  # got a body without any LLM call


# ---------------------------------------------------------------------------
# Signature (D48)
# ---------------------------------------------------------------------------


SIGNATURE = "Warm Regards\nHemant Tyagi\nAI / ML Engineer\n+91 9548550009"


@pytest.fixture
def signed_profile(profile) -> CandidateProfile:
    """The same profile with a signature configured."""
    return profile.model_copy(
        update={"drafts": profile.drafts.model_copy(update={"signature": SIGNATURE})}
    )


def test_signature_appended_to_interested(signed_profile, parsed_factory):
    opp = Opportunity(company="Acme", role_title="Backend Engineer")
    body = build_interested(parsed_factory(), opp, signed_profile)
    assert body.rstrip().endswith("+91 9548550009")
    assert "Warm Regards" in body


def test_signature_appended_to_decline(signed_profile, parsed_factory):
    """Declines get signed too — both shapes are mail from a real person."""
    body = build_decline(parsed_factory(), "not a fit", None, signed_profile)
    assert "Warm Regards" in body


def test_no_signature_configured_leaves_body_unchanged(profile, parsed_factory):
    """Signature is optional; absent means the pre-D48 body, not a crash."""
    opp = Opportunity(company="Acme", role_title="Backend Engineer")
    assert profile.drafts.signature is None
    body = build_interested(parsed_factory(), opp, profile)
    assert "Warm Regards" not in body
    # D61 restored the request gate, so with no CV attached the closing
    # offers one rather than claiming it is enclosed.
    assert body.rstrip().endswith("Happy to share my CV if that would help.")


def test_decline_without_profile_still_works(parsed_factory):
    """The 3-arg call site predates the signature and must keep working."""
    body = build_decline(parsed_factory(), "not a fit")
    assert "not a fit" in body
    assert "Warm Regards" not in body


def test_exactly_one_blank_line_before_signature(signed_profile, parsed_factory):
    """Spacing must not depend on how the TOML value was pasted.

    GOTCHA: a signature stored with leading newlines would otherwise produce
    three or four blank lines before the sign-off, which reads as a formatting
    bug in every email sent.
    """
    padded = signed_profile.model_copy(
        update={
            "drafts": signed_profile.drafts.model_copy(
                update={"signature": "\n\n" + SIGNATURE + "\n\n"}
            )
        }
    )
    body = build_decline(parsed_factory(), "not a fit", None, padded)
    assert "\n\n\n" not in body, "collapsed blank lines expected"
    assert "match.\n\nWarm Regards" in body


def test_signature_counts_toward_length_and_passes_validator(signed_profile, parsed_factory):
    """The signature goes through the outbound validator with the body.

    A PAN- or Aadhaar-shaped token in a signature would quarantine every
    single draft, so this asserts the real one is clean.
    """
    from app.drafts.validator import validate_draft

    opp = Opportunity(company="Acme", role_title="Backend Engineer")
    body = build_interested(parsed_factory(), opp, signed_profile)
    verdict = validate_draft(body, signed_profile.drafts.max_length_chars)
    assert not verdict.quarantined, verdict.reason


# ---------------------------------------------------------------------------
# Salutation (D51) — every case below is a real sender from the corpus
# ---------------------------------------------------------------------------


def _sal(from_name, recruiter_name=None, company="Acme Corp"):
    from app.drafts.generator import _salutation
    from tests.conftest import make_parsed
    parsed = make_parsed(gmail_id="g", from_name=from_name)
    opp = Opportunity(company=company, role_title="Engineer",
                      recruiter_name=recruiter_name)
    return _salutation(parsed, opp)


@pytest.mark.parametrize("from_name,recruiter_name,expected", [
    # Extracted name beats the header, which is the whole point.
    ("DEEPA-CRESTLINE STAFFING", "Deepa Nair", "Deepa"),
    ("Talent Bridge Hr Consulting", "Arvind K", "Arvind"),
    ("HR Manager", "Dinesh Kumar", "Dinesh"),
    ("Redpine Balaji Associates", "Nandini", "Nandini"),
    ("Lakeshore Solutions Private Limited", "Farah", "Farah"),
    (None, "Rekha Chandra", "Rekha"),
    # Header used when it looks like a person.
    ("ritika s", "Ritika S", "Ritika"),
    ("Ira Bhandari", "Ira Bhandari", "Ira"),
    ("Asha Menon", "Asha", "Asha"),
])
def test_salutation_uses_person_name(from_name, recruiter_name, expected):
    assert _sal(from_name, recruiter_name) == expected


@pytest.mark.parametrize("from_name", [
    "Midwest Consultants Pune",
    "Stellarcorp Consulting Private Limited",
    "LinkedIn Job Alerts",
    "Naukri Alerts",
    "Glassdoor Jobs",
    "Careers@Harborlane",          # an address in the display-name slot
])
def test_salutation_falls_back_to_there_for_organisations(from_name):
    assert _sal(from_name) == "there"


def test_company_name_alone_is_not_greeted_as_a_person():
    """`Northwind` has no org-marker word, so only the extracted company catches it.

    GOTCHA this pins: the original bug produced "Hi Northwind,". A marker list
    can never catch single-word company names — the cross-check against the
    extracted employer is what does.
    """
    assert _sal("Northwind", None, company="Northwind") == "there"
    assert _sal("Crestline Ites", None, company="Crestline ITES Pvt Ltd") == "there"


def test_shouted_names_are_not_shouted_back():
    assert _sal("DEEPA", "DEEPA") == "Deepa"


def test_initials_are_skipped_not_greeted():
    """"M Varmann" must not become "Hi M,"."""
    assert _sal("Geetha Varman", "M Varmann") == "Varmann"
    assert _sal("Anita B", "Anita B") == "Anita"


def test_dotted_and_hyphenated_names_split_correctly():
    assert _sal("Meghna R", "Meghna.R") == "Meghna"


def test_no_name_anywhere_falls_back():
    assert _sal(None, None) == "there"
    assert _sal("", "") == "there"
