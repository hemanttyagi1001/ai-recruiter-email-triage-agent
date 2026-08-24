"""
Pydantic v2 schemas the LLM is asked to return. These serve two masters:

1. They generate the JSON schema fed to Azure OpenAI's structured-output
   API. When strict mode is on, the API's decoder is grammar-constrained by
   this schema — the model literally cannot emit malformed JSON or wrong
   field types. That gives us free structural guarantees.

2. They validate the model's output on the Python side. Structural
   guarantees do not cover *semantic* rules: "ctc_max >= ctc_min", "return
   null when the email doesn't state a value." Those live as Pydantic
   validators here.

The Opportunity model is the contract for the extract node's output. Any
change to fields flows both into the DB (via db/models.py) and into the
extractor's prompt (which describes each field).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# WHY Literal[...] over StrEnum here: the openai SDK's beta.chat.completions.
# parse converts Pydantic types to strict JSON schema. Literal unions become
# a JSON schema `enum`, which strict mode enforces at decode time — the
# model literally cannot emit a value outside this list. A Python StrEnum
# would work too but Literal is one fewer indirection.
CategoryValue = Literal[
    "new_role_pitch",
    "followup_existing_thread",
    "interview_scheduling",
    "document_request",
    "paid_placement_solicitation",
    "resume_service_solicitation",
    "not_recruitment",
]


class ClassificationResult(BaseModel):
    # CONCEPT: extra="forbid" means "the model may not add fields we didn't
    # ask for." Combined with strict mode on the API side, this closes the
    # door on the model "helpfully" adding a suggestions field we then have
    # to notice and discard.
    model_config = ConfigDict(extra="forbid")

    category: CategoryValue
    confidence: float = Field(ge=0.0, le=1.0)


WorkModelValue = Literal["onsite", "hybrid", "remote"]
EmploymentTypeValue = Literal["permanent", "c2h", "contract"]


class Opportunity(BaseModel):
    """Structured extraction of a recruiter email.

    Every field is Optional. Optional here means "the email did not
    explicitly state this" — not "we couldn't parse it." The extractor
    prompt instructs the model to return null rather than infer.
    """

    model_config = ConfigDict(extra="forbid")

    company: str | None = None
    end_client: str | None = None
    role_title: str | None = None
    location: str | None = None
    work_model: WorkModelValue | None = None
    employment_type: EmploymentTypeValue | None = None
    # Salary normalised to LPA (lakhs per annum, ₹100,000/year). See D9.
    ctc_min_lpa: float | None = None
    ctc_max_lpa: float | None = None
    notice_period: str | None = None
    recruiter_name: str | None = None
    recruiter_email: str | None = None
    recruiter_phone: str | None = None
    source_platform: str | None = None
    jd_text: str | None = None

    @model_validator(mode="after")
    def _ctc_range(self) -> "Opportunity":
        # WHY at the model level, not per-field: the invariant depends on
        # *two* fields. Field-level validators can't see siblings.
        # CONCEPT: this is exactly the class of check strict JSON schema
        # cannot express. Structural mode enforces "both are numbers or
        # null"; only Pydantic can enforce "if both present, max >= min."
        # Its failure feeds the retry loop in extract.py.
        if self.ctc_min_lpa is not None and self.ctc_max_lpa is not None:
            if self.ctc_max_lpa < self.ctc_min_lpa:
                raise ValueError(
                    f"ctc_max_lpa ({self.ctc_max_lpa}) must be >= "
                    f"ctc_min_lpa ({self.ctc_min_lpa})"
                )
        return self


class FitScore(BaseModel):
    """Fit scorer output. See app/scoring/fit.py for the prompt and CONCEPT
    comments on why abstention (`uncertain=True`) is a first-class field."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=100)
    rationale: str = Field(max_length=400)
    # CONCEPT: `uncertain` is not just documentation for humans — the graph
    # reads it as a routing signal. Downstream, uncertain=True is treated
    # as "route to human review, no auto-draft," entirely independent of
    # the numeric score. That structural respect for the abstain signal is
    # what makes it meaningful; a schema field the model can set but nothing
    # reads is theatre.
    uncertain: bool
