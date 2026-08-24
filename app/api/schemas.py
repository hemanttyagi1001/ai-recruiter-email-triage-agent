"""
Request/response models for the approval API. These are separate Pydantic
classes from the LLM schemas — endpoints exchange domain shapes, not
graph-internal shapes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PendingSummary(BaseModel):
    """One row in GET /pending."""
    model_config = ConfigDict(extra="forbid")

    thread_id: str            # = message_id
    subject: str
    from_email: str
    from_name: str | None
    received_at: datetime
    category: str | None
    draft_type: str
    fit_score: int | None
    fit_uncertain: bool | None
    rule_name: str | None
    quarantined: bool
    created_at: datetime


class OpportunityView(BaseModel):
    """Extracted opportunity fields — mirrors the DB row minus id/timestamps."""
    model_config = ConfigDict(extra="forbid")

    company: str | None
    end_client: str | None
    role_title: str | None
    location: str | None
    work_model: str | None
    employment_type: str | None
    ctc_min_lpa: float | None
    ctc_max_lpa: float | None
    notice_period: str | None
    recruiter_name: str | None
    recruiter_email: str | None
    recruiter_phone: str | None
    source_platform: str | None
    jd_text: str | None


class NodeEventView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str
    at: datetime
    duration_ms: int
    outcome: str | None


class PendingDetail(BaseModel):
    """Full detail for GET /pending/{thread_id}."""
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    subject: str
    from_email: str
    from_name: str | None
    received_at: datetime
    body_text: str
    category: str | None
    opportunity: OpportunityView | None
    draft_type: str
    draft_body: str
    quarantined: bool
    quarantine_reason: str | None
    rule_name: str | None
    rule_reason: str | None
    fit_score: int | None
    fit_rationale: str | None
    fit_uncertain: bool | None
    events: list[NodeEventView]
    created_at: datetime


class ApproveRequest(BaseModel):
    """POST /pending/{thread_id}/approve body.

    edited_body: if the human tweaked the draft. Trust the human — no
    re-validation applied (per Phase 2 Q5 answer). Just forwarded to act.
    note: optional annotation stored on drafts.approval_reason.
    """
    model_config = ConfigDict(extra="forbid")
    edited_body: str | None = None
    note: str | None = None


class RejectRequest(BaseModel):
    """POST /pending/{thread_id}/reject body. Reason is required."""
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1)


class ResolutionResponse(BaseModel):
    """Response after approve/reject."""
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    final_status: str
    gmail_draft_id: str | None = None
