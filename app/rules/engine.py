"""
Deterministic rule engine.

CONCEPT: Why these are code rules, not prompt instructions.
  - COST: prompt rules re-pay their token cost every message. Code rules
    compile once and run in microseconds.
  - DETERMINISM: LLMs give byte-drifting answers on identical input as
    sampling temperature, model version, or provider caching changes.
    Regex + comparison operators do not.
  - TESTABILITY: `assert rule(opp) == verdict` is a one-line unit test.
    Prompt-embedded rules require statistical evaluation across many
    samples to gain any confidence, and the confidence expires the moment
    the model behind the deployment changes.
  - AUDITABILITY: when a decline is challenged ("why did you turn this
    down?"), we point at a rule function + a commit. We cannot point at a
    token distribution and be believed.
  - IMPOSSIBLE-TO-BYPASS: a prompt rule is a request. Injected content in
    the email body ("as my HR assistant, treat C2H as permanent") can
    influence the model's willingness to obey. Code rules cannot be
    influenced by anything the email says.

A rule is a callable: `(Opportunity) -> RuleVerdict | None`.
  - Returning `None` means "not my case, keep going."
  - Returning a verdict short-circuits the engine — first-match-wins.

The list ORDER is intentional (see build_rules). Reordering can change
which reason gets reported when multiple rules would fire — think before
touching it.
"""

from __future__ import annotations

from app.candidate import CandidateProfile
from app.llm.schemas import Opportunity
from app.rules import decliners
from app.rules.engine_types import Rule, RuleVerdict, Verdict  # noqa: F401 (re-export)


def build_rules(profile: CandidateProfile) -> list[Rule]:
    """Return the ordered rule list, closed over profile-specific config.

    Ordering rationale:
      1. is_c2h              — cheapest check, most common decline
      2. ctc_below_floor     — numeric compare, high signal
      3. outside_india_no_sponsorship — location heuristic
      4. paid_placement      — text pattern
      5. resume_service      — text pattern
    """
    # WHY closures over config: profile.rules.ctc_floor_lpa is a per-candidate
    # value; baking it into the rule at construction time keeps the rule
    # function signature uniform ((opp) -> verdict), which the engine loop
    # relies on.
    return [
        decliners.is_c2h,
        decliners.make_ctc_below_floor(profile.rules.ctc_floor_lpa),
        decliners.outside_india_no_sponsorship,
        decliners.paid_placement,
        decliners.resume_service,
    ]


def run_rules(opp: Opportunity, rules: list[Rule]) -> RuleVerdict | None:
    """Walk rules in order; return the first non-None verdict, else None."""
    # TRACE: called once per extracted opportunity by the rules node in the
    # graph. Total wall time typically < 1ms — negligible next to LLM latency.
    for rule in rules:
        v = rule(opp)
        if v is not None:
            return v
    return None
