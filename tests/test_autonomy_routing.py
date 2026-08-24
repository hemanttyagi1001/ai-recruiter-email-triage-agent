"""
Tests for _route_after_validate — the autonomy gate.

HISTORY, because this file used to assert the opposite. Under D33 (Phase 5)
autonomy was the intersection of five conditions: rule-fired, decline-only,
validator-clean, no fit score, no duplicate flag. D45 (Phase 6) removed four
of them at the operator's explicit direction — declines AND interested replies
now send without a human.

Exactly one gate survived, and it is the point of this file: a QUARANTINED
draft never auto-sends, in any mode. That is not a preference the config can
override. D14 makes quarantine a code rule — the PAN/Aadhaar scan and the
length cap — and "this text must not leave the building" is categorically
different from "we prefer a human to look first".

The matrix is mode x state because the routing now depends on both, and the
interesting property is which combinations can reach `auto_send`.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.db.models import DraftType
from app.drafts.validator import ValidationVerdict
from app.llm.schemas import FitScore
from app.pipeline.graph import _route_after_validate
from app.rules.engine_types import RuleVerdict


@pytest.fixture
def auto_send_mode(monkeypatch):
    """Set settings.auto_send_mode for one test.

    WHY monkeypatch the settings object rather than the env var: `settings` is
    instantiated once at import time (see app/config.py), so mutating
    AZURE-style env vars after import changes nothing. Patching the attribute
    is the only thing that affects an already-loaded process — which is also
    true in production, and worth knowing: changing AUTO_SEND_MODE requires a
    restart, unlike the kill switch which is read from the DB every send.
    """
    def _set(value: str) -> None:
        monkeypatch.setattr(settings, "auto_send_mode", value)
    return _set


def _rv() -> RuleVerdict:
    return RuleVerdict(rule_name="c2h", verdict="decline", reason="C2H roles")


def _fit_score(score: int = 40) -> FitScore:
    return FitScore(score=score, rationale="test", uncertain=False)


def _clean() -> ValidationVerdict:
    return ValidationVerdict(quarantined=False, reason=None, body_text="body")


def _quarantined() -> ValidationVerdict:
    return ValidationVerdict(quarantined=True, reason="PII", body_text="body")


# States that D33 used to block and D45 now permits. Each one is a behaviour
# change worth naming, so a future reader can see exactly what was traded away.
NOW_ELIGIBLE = [
    (
        "rule_fired_decline_still_eligible",
        {"rule_verdict": _rv(), "draft_type": DraftType.DECLINE,
         "draft_validation": _clean()},
    ),
    (
        "no_rule_verdict_now_eligible",  # was D33 gate 1
        {"rule_verdict": None, "draft_type": DraftType.DECLINE,
         "draft_validation": _clean()},
    ),
    (
        "interested_draft_now_eligible",  # was D33 gate 2 — the big one
        {"rule_verdict": _rv(), "draft_type": DraftType.INTERESTED,
         "draft_validation": _clean()},
    ),
    (
        "fit_score_present_now_eligible",  # was D33 gate 4
        {"rule_verdict": _rv(), "draft_type": DraftType.DECLINE,
         "draft_validation": _clean(), "fit_score_result": _fit_score()},
    ),
    (
        "duplicate_flags_now_eligible",  # was D33 gate 5
        {"rule_verdict": _rv(), "draft_type": DraftType.DECLINE,
         "draft_validation": _clean(), "duplicate_candidates": [object()]},
    ),
    (
        "missing_validation_treated_as_not_quarantined",
        {"rule_verdict": _rv(), "draft_type": DraftType.DECLINE},
    ),
]


@pytest.mark.parametrize("armed_mode", ["dry_run", "on"])
@pytest.mark.parametrize("label,state", NOW_ELIGIBLE, ids=[c[0] for c in NOW_ELIGIBLE])
def test_armed_modes_route_to_auto_send(label, state, armed_mode, auto_send_mode):
    """dry_run and on must route identically.

    A dry run that took a different path would measure the wrong thing — the
    whole value of dry_run is that it answers "what would happen if this were
    armed", so the routing has to be the same and only the Gmail call differs.
    """
    auto_send_mode(armed_mode)
    assert _route_after_validate(state) == "auto_send", f"case={label}"


@pytest.mark.parametrize("label,state", NOW_ELIGIBLE, ids=[c[0] for c in NOW_ELIGIBLE])
def test_mode_off_always_routes_to_human(label, state, auto_send_mode):
    """off restores the Phase 5 behaviour for every state."""
    auto_send_mode("off")
    assert _route_after_validate(state) == "persist_pending", f"case={label}"


@pytest.mark.parametrize("mode", ["off", "dry_run", "on"])
def test_quarantined_never_auto_sends_in_any_mode(mode, auto_send_mode):
    """The one gate config cannot open.

    If this test ever fails, a draft that tripped the PII scan or the length
    cap became eligible to be emailed to a recruiter. There is no mode, flag,
    or profile setting that is supposed to make that possible.
    """
    auto_send_mode(mode)
    state = {
        "rule_verdict": _rv(),
        "draft_type": DraftType.DECLINE,
        "draft_validation": _quarantined(),
    }
    assert _route_after_validate(state) == "persist_pending"


# ---------------------------------------------------------------------------
# draft mode routes like the other armed modes (D49)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,state", NOW_ELIGIBLE, ids=[c[0] for c in NOW_ELIGIBLE])
def test_draft_mode_routes_to_auto_send(label, state, auto_send_mode):
    """`draft` must reach auto_send, same as dry_run and on.

    The soak argument depends on it: days spent in `draft` are only evidence
    about `on` if both take an identical path and differ solely in which Gmail
    call fires at the end.
    """
    auto_send_mode("draft")
    assert _route_after_validate(state) == "auto_send", f"case={label}"


def test_draft_mode_still_respects_quarantine(auto_send_mode):
    """Quarantine outranks every mode, drafting included.

    A draft carrying a PAN or Aadhaar token is one careless click away from
    being sent, so the validator's veto has to hold here too.
    """
    auto_send_mode("draft")
    state = {
        "rule_verdict": _rv(),
        "draft_type": DraftType.DECLINE,
        "draft_validation": _quarantined(),
    }
    assert _route_after_validate(state) == "persist_pending"


# ---------------------------------------------------------------------------
# Abstentions now produce a clarifying draft (D54)
# ---------------------------------------------------------------------------


def test_uncertain_score_routes_to_draft_not_terminal():
    """An abstention is answerable; a missing score is not.

    These two shared a branch until D54, which left 9 of 10 real HR emails
    with no reply at all. `uncertain` means the JD was too vague to rate — the
    right response is to ask for what it left out, not silence.
    """
    from app.pipeline.graph import _route_after_score

    assert _route_after_score({"fit_score_result": _fit_score(50)}) == "draft"
    uncertain = FitScore(score=0, rationale="too vague", uncertain=True)
    assert _route_after_score({"fit_score_result": uncertain}) == "draft"


def test_missing_score_still_terminates():
    """No result at all means the scorer broke — there is nothing to say."""
    from app.pipeline.graph import _route_after_score

    assert _route_after_score({}) == "persist_terminal"
    assert _route_after_score({"fit_score_result": None}) == "persist_terminal"
