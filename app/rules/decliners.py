"""
The five decliner rules. Each is a plain function of Opportunity.

Rules 4 and 5 (paid_placement, resume_service) may look redundant with the
classifier categories `paid_placement_solicitation` and
`resume_service_solicitation` — messages in those categories never reach the
rules engine (they short-circuit to persist after classification). The rules
are here as DEFENCE IN DEPTH: if the classifier misfires (paid-placement
language buried in paragraph 4 of a "great opportunity" pitch), the rule
engine sees the full extracted body and still catches it before we generate
an "interested" reply into a scam.
"""

from __future__ import annotations

import re

from app.llm.schemas import Opportunity
from app.rules.engine_types import Rule, RuleVerdict  # forward-declared below

# NOTE: to avoid a circular import (engine.py imports this module for the
# rule functions, this module needs RuleVerdict from engine.py), we define
# the shared types in a tiny engine_types module below rather than in
# engine.py itself. See engine_types.py.


# --- Rule 1: employment_type is C2H ---


def is_c2h(opp: Opportunity) -> RuleVerdict | None:
    if opp.employment_type == "c2h":
        return RuleVerdict(
            rule_name="c2h",
            verdict="decline",
            reason="C2H (contract-to-hire) arrangements are not a fit",
        )
    return None


# --- Rule 2: ctc_max below the configured floor ---


def make_ctc_below_floor(floor_lpa: float) -> Rule:
    """Factory: closes the floor value into a rule function."""

    def ctc_below_floor(opp: Opportunity) -> RuleVerdict | None:
        # WHY the None guard first: unstated CTC is not a decline. Many
        # legitimate roles omit the number and expect negotiation. Only an
        # explicitly-stated ceiling below the floor should trigger.
        if opp.ctc_max_lpa is None:
            return None
        if opp.ctc_max_lpa < floor_lpa:
            return RuleVerdict(
                rule_name="ctc_below_floor",
                verdict="decline",
                reason=(
                    f"stated CTC ceiling of {opp.ctc_max_lpa} LPA is below the "
                    f"{floor_lpa} LPA floor"
                ),
            )
        return None

    return ctc_below_floor


# --- Rule 3: outside India, no sponsorship indicated ---

# CONCEPT: this whitelist is deliberately dumb string-matching, not a
# geo library. Reasons: (a) location strings in recruiter mail are messy
# ("Bangalore/Remote", "Onsite - Bengaluru India", "London or Bangalore"),
# (b) we care about a binary "is any India signal present," not real
# geocoding, (c) missed matches bias toward decline, which is the safe
# direction for a candidate not currently open to relocation.
# See DECISIONS.md D16.
INDIA_TOKENS: frozenset[str] = frozenset(
    {
        # country
        "india",
        # cities and metros (metros first)
        "bangalore", "bengaluru", "mumbai", "delhi", "gurgaon", "gurugram",
        "noida", "hyderabad", "pune", "chennai", "kolkata", "ahmedabad",
        "jaipur", "chandigarh", "kochi", "trivandrum", "coimbatore", "indore",
        "lucknow", "bhopal", "nagpur", "vadodara", "surat", "faridabad",
        "ghaziabad", "mysore", "mysuru", "vizag", "visakhapatnam",
        # region shorthands
        "ncr", "hinjewadi", "whitefield", "electronic city", "hitech city",
        "cyberabad", "gachibowli",
        # states
        "karnataka", "maharashtra", "telangana", "tamil nadu", "tamilnadu",
        "kerala", "haryana", "uttar pradesh", "west bengal", "gujarat",
        "rajasthan", "punjab", "andhra pradesh", "odisha", "assam",
    }
)

SPONSORSHIP_PATTERN = re.compile(
    r"\b(sponsor(?:ship)?|visa|h[- ]?1[- ]?b|work permit|relocation package|"
    r"immigration support|green card|permanent residency|pr\s+support)\b",
    re.IGNORECASE,
)


def outside_india_no_sponsorship(opp: Opportunity) -> RuleVerdict | None:
    # No location stated → can't judge → don't decline on this rule.
    if not opp.location:
        return None
    # Remote work is location-agnostic; the fact that the role is nominally
    # HQ'd abroad doesn't require relocation.
    if opp.work_model == "remote":
        return None

    loc = opp.location.lower()
    # Any India token present → India-based → pass.
    if any(token in loc for token in INDIA_TOKENS):
        return None

    # Location is outside India. If the JD explicitly mentions sponsorship,
    # let the scorer / human make the call.
    jd = (opp.jd_text or "").lower()
    if SPONSORSHIP_PATTERN.search(jd):
        return None

    return RuleVerdict(
        rule_name="outside_india_no_sponsorship",
        verdict="decline",
        reason=(
            f"role location {opp.location!r} is outside India with no "
            f"relocation or sponsorship mentioned"
        ),
    )


# --- Rule 4: paid placement / candidate-pays language ---

PAID_PLACEMENT_PATTERN = re.compile(
    r"\b(processing fee|registration fee|refundable deposit|candidate pays?|"
    r"placement fee|consultation charge|charges? apply|pay(?:ment)?\s+required|"
    r"one[- ]time fee|nominal fee|training fee|onboarding fee)\b",
    re.IGNORECASE,
)


def paid_placement(opp: Opportunity) -> RuleVerdict | None:
    haystack = " ".join(filter(None, [opp.jd_text, opp.role_title, opp.company]))
    m = PAID_PLACEMENT_PATTERN.search(haystack)
    if m:
        return RuleVerdict(
            rule_name="paid_placement",
            verdict="decline",
            reason=f"content indicates paid-placement / candidate-side fee "
                   f"(matched: {m.group(0)!r})",
        )
    return None


# --- Rule 5: resume writing / profile enhancement pitch ---

RESUME_SERVICE_PATTERN = re.compile(
    r"\b(resume writing|resume revamp|resume makeover|profile enhancement|"
    r"profile revamp|cv writing|cv makeover|linkedin optimi[sz]ation|"
    r"linkedin profile writing|professional cv|ats[- ]friendly resume)\b",
    re.IGNORECASE,
)


def resume_service(opp: Opportunity) -> RuleVerdict | None:
    haystack = " ".join(filter(None, [opp.jd_text, opp.role_title, opp.company]))
    m = RESUME_SERVICE_PATTERN.search(haystack)
    if m:
        return RuleVerdict(
            rule_name="resume_service",
            verdict="decline",
            reason=f"content pitches resume/profile services rather than a job "
                   f"(matched: {m.group(0)!r})",
        )
    return None
