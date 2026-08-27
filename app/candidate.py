"""
Candidate profile loader — reads candidate.toml and validates it into a
Pydantic model. This is the config that fills the "interested" draft
template, provides the CTC floor for the rules engine, and gives the fit
scorer its notion of "who this candidate is."

WHY separate from app/config.py: config.py holds infra secrets (Azure keys,
DB URL) that live in env vars. This holds personal profile data that lives
in a TOML file. Same lifecycle (load once at boot) but different source and
different secrecy posture — profile isn't a secret, but salary and location
data is personal enough to keep out of env vars where a `printenv` could
leak it.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)


# The literal rendered wherever a profile value is absent. Single constant so
# the draft template, the fit-scorer prompt, and the startup warning can never
# drift into disagreeing about what "unfilled" looks like.
NA = "NA"


def render(value: object | None) -> str:
    """Render a profile value for a prompt or a draft, or NA if unset.

    CONCEPT: absent is not the same as zero or empty. A candidate who has not
    filled in `current_ctc_lpa` is not a candidate on 0 LPA, and a scorer told
    "Current CTC: 0 LPA" would draw a confidently wrong conclusion. Rendering
    an explicit NA keeps "we don't know" distinguishable from a real value —
    the same reasoning the extractor already applies to email fields, where a
    fact the source doesn't state becomes null rather than a guess.

    Floats lose a pointless trailing .0 (14.0 -> "14") because these land in
    prose, where "14 years" reads and "14.0 years" does not.
    """
    if value is None:
        return NA
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else str(value)
    return str(value)


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # WHY only `name` stays required: it is the one field that identifies whose
    # profile this is, it appears in no outbound text, and leaving it blank
    # signals a file that was never filled in at all rather than one still
    # being completed.
    name: str

    # CONCEPT: optional-with-NA, and what it costs.
    #   Every field below feeds two places — the "interested" draft template
    #   and the fit-scorer's system prompt. Making them optional means a
    #   half-filled profile boots and runs instead of failing at import.
    # GOTCHA: that convenience has a sharp edge under autonomous sending. An
    #   "interested" reply generated from an empty profile is a real email to a
    #   real recruiter reading "I have NA years total, NA LPA current". Absent
    #   values do not block a send; they just render. `missing_fields()` exists
    #   so startup can say loudly which ones are still unfilled. See D46.
    total_years: float | None = None
    relevant_years: float | None = None
    stack: str | None = None
    current_ctc_lpa: float | None = None
    expected_ctc_lpa: float | None = None
    notice_period: str | None = None
    current_location: str | None = None
    preferred_location: str | None = None
    employment_status: str | None = None
    # D67: asked for by name on nearly every Indian screening form, and by
    # nothing else in the system. They exist so the questionnaire reply can
    # answer the whole form instead of most of it — a form returned with two
    # blanks reads as evasive on the two questions a recruiter is screening on.
    # GOTCHA: still Optional. An unset value is omitted from the answer block,
    # never guessed — the standing rule that a field the source does not state
    # is null, not an invention.
    native_location: str | None = None
    reason_for_job_change: str | None = None

    @field_validator(
        "stack", "notice_period", "current_location",
        "preferred_location", "employment_status",
        "native_location", "reason_for_job_change",
        mode="before",
    )
    @classmethod
    def _unfilled_placeholder_is_none(cls, v: object) -> object:
        """Treat a leftover FILL-ME placeholder as an unset field.

        WHY this exists: candidate.toml.example ships every optional field
        commented out, so the normal path to "unfilled" is simply an absent
        line. But the obvious way to edit a config is to uncomment everything
        first and fill it in as you go — which leaves real FILL-ME strings in
        a file that parses perfectly. Without this, the agent would email a
        recruiter "I have FILL-ME years of experience".
        GOTCHA: string fields only. A numeric placeholder cannot be
        distinguished from a real number — `total_years = 0` is a valid
        answer, just a wrong one — which is why the example comments the
        numeric lines out instead of seeding them with a sentinel.
        """
        if isinstance(v, str) and v.strip().upper().startswith("FILL-ME"):
            return None
        return v

    def missing_fields(self) -> list[str]:
        """Names of profile fields still unset, in declaration order.

        Declaration order rather than sorted: it matches the order of the
        commented placeholders in candidate.toml.example, so the warning reads
        as a checklist against the file the user is editing.
        """
        return [
            field
            for field in self.__class__.model_fields
            if field != "name" and getattr(self, field) is None
        ]


class _Rules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ctc_floor_lpa: float = Field(gt=0)
    # WHY a toggle rather than deleting the rule: "entertain C2H" is a stance
    # that tracks the job market and the candidate's appetite, and it flips
    # back. A config flag keeps the rule, its reason string and its test
    # alive so re-arming it is one word, not a revert.
    # GOTCHA: defaults to False, i.e. C2H still declines, so an existing
    # candidate.toml that says nothing keeps the pre-D65 behaviour. Opting
    # into risk should require typing something.
    # TRACE: read once by build_rules at graph-construction time, not per
    # message — flipping it needs a restart, unlike the kill switch.
    allow_c2h: bool = False


class _Scoring(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fit_threshold: int = Field(ge=0, le=100)


class _Drafts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_length_chars: int = Field(gt=0)

    # CONCEPT: why the agent has to emit its own signature, when the original
    # design deliberately did not.
    #   The templates were written to stop before the sign-off, on the
    #   reasoning that a mail client appends the user's configured signature.
    #   That holds when a human composes in the Gmail web UI. It does NOT hold
    #   for anything this agent produces: a draft created through
    #   drafts.create carries exactly the MIME body we supply, and an
    #   autonomously sent reply through messages.send never passes through the
    #   UI at all. Under AUTO_SEND_MODE=on the original assumption produces
    #   unsigned mail to recruiters. See D48.
    # WHY it lives here rather than in a template file: it is personal contact
    # data — phone, email, LinkedIn — and candidate.toml is already the
    # gitignored home for exactly that, already mounted into the container.
    # A separate signature.txt would mean a second secret file and a second
    # bind mount for one string.
    # GOTCHA: this text is appended BEFORE the outbound validator runs, so a
    # signature containing a PAN- or Aadhaar-shaped token would quarantine
    # every draft. Verified clean for the current one; re-check after editing.
    signature: str | None = None


class _Dedup(BaseModel):
    # Phase 4 — dedup knobs. See candidate.toml.example for prose on each.
    # Defaults let profiles from earlier phases load without editing; a
    # missing [dedup] section falls back to the same defaults the example
    # ships with.
    model_config = ConfigDict(extra="forbid")
    lookback_days: int = Field(default=60, gt=0)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_jd_chars: int = Field(default=100, ge=0)
    max_candidates_returned: int = Field(default=5, gt=0)


class CandidateProfile(BaseModel):
    """The whole candidate.toml, parsed and validated."""

    model_config = ConfigDict(extra="forbid")
    candidate: _Candidate
    rules: _Rules
    scoring: _Scoring
    drafts: _Drafts
    # WHY default_factory not Field(default=_Dedup()): a shared default
    # instance across profiles is fine here (immutable Pydantic model),
    # but default_factory is the idiomatic way to say "give each new
    # profile its own defaulted _Dedup" if we ever add mutable fields.
    dedup: _Dedup = Field(default_factory=_Dedup)


def load_profile(path: Path = Path("candidate.toml")) -> CandidateProfile:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy candidate.toml.example to candidate.toml "
            f"and fill in your profile before running the pipeline."
        )
    # WHY tomllib (stdlib) over tomlkit or third-party: read-only load is all
    # we need. tomllib ships with Python 3.11+ — zero extra dependency.
    with path.open("rb") as f:
        raw = tomllib.load(f)
    profile = CandidateProfile.model_validate(raw)

    # WHY warn at load rather than at draft time: by the time a draft renders
    # NA, the message is mid-graph and the log line is buried among LLM calls.
    # Here it appears once, at boot, next to the file it refers to — the only
    # moment the operator is actually in a position to go and fix it.
    missing = profile.candidate.missing_fields()
    if missing:
        log.warning(
            "candidate.toml has %d unfilled field(s): %s. These render as %r "
            "in outbound drafts and in the fit-scorer prompt. Fill the "
            "placeholders in %s to replace them.",
            len(missing), ", ".join(missing), NA, path,
        )
    return profile
