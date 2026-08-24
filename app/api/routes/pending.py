"""
The four approval endpoints:

    GET  /pending
    GET  /pending/{thread_id}
    POST /pending/{thread_id}/approve
    POST /pending/{thread_id}/reject

CONCEPT: two data sources per endpoint.
  - The DB (via SQLAlchemy) is the domain view — "which messages are
    awaiting approval, with what draft, at what fit score." Fast, indexed,
    good for list/detail.
  - The checkpointer (via graph.get_state) is the state-machine view —
    "what's in the LangGraph state right now, including the events log."
    We only reach for it in /pending/{thread_id} where we need the events.

Approve/reject drives the graph forward by injecting an update via
Command; LangGraph resumes from the interrupt, runs act, runs
persist_final, and returns terminal state.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.types import Command
from sqlalchemy import select

from app.api.deps import get_graph
from app.api.schemas import (
    ApproveRequest,
    NodeEventView,
    OpportunityView,
    PendingDetail,
    PendingSummary,
    RejectRequest,
    ResolutionResponse,
)
from app.db.engine import session_scope
from app.db.models import Draft, DraftStatus, Message, MessageStatus, Opportunity

log = logging.getLogger(__name__)

router = APIRouter(prefix="/pending", tags=["pending"])


# =============================================================================
# GET /pending — list all threads awaiting human approval
# =============================================================================


@router.get("", response_model=list[PendingSummary])
def list_pending() -> list[PendingSummary]:
    """Return everything with messages.status = awaiting_approval."""
    # Domain query only — no checkpointer touch. The summary shape is
    # what the DB already knows.
    with session_scope() as s:
        rows = s.execute(
            select(Message, Draft)
            .join(Draft, Draft.message_id == Message.message_id)
            .where(Message.status == MessageStatus.AWAITING_APPROVAL)
            .order_by(Message.received_at.desc())
        ).all()

        return [
            PendingSummary(
                thread_id=msg.message_id,
                subject=msg.subject,
                from_email=msg.from_email,
                from_name=msg.from_name,
                received_at=msg.received_at,
                category=msg.category,
                draft_type=draft.draft_type,
                fit_score=draft.fit_score,
                fit_uncertain=draft.fit_uncertain,
                rule_name=draft.rule_name,
                quarantined=draft.quarantine_reason is not None,
                created_at=draft.created_at,
            )
            for msg, draft in rows
        ]


# =============================================================================
# GET /pending/{thread_id} — full detail including events from the checkpointer
# =============================================================================


@router.get("/{thread_id}", response_model=PendingDetail)
def get_pending_detail(thread_id: str, graph=Depends(get_graph)) -> PendingDetail:
    with session_scope() as s:
        msg = _get_awaiting_message(s, thread_id)
        draft = s.query(Draft).filter_by(message_id=thread_id).one_or_none()
        opp = s.query(Opportunity).filter_by(message_id=thread_id).one_or_none()

        if draft is None:
            # Awaiting approval but no draft row — should never happen.
            raise HTTPException(500, f"invariant broken: message {thread_id} has no draft row")

        # Fetch the events list from the checkpointer state. This is the
        # only place we reach into LangGraph internals from the API.
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        events_raw = snapshot.values.get("events", []) if snapshot else []

        return PendingDetail(
            thread_id=msg.message_id,
            subject=msg.subject,
            from_email=msg.from_email,
            from_name=msg.from_name,
            received_at=msg.received_at,
            body_text=msg.body_text,
            category=msg.category,
            opportunity=OpportunityView(
                company=opp.company, end_client=opp.end_client, role_title=opp.role_title,
                location=opp.location, work_model=opp.work_model,
                employment_type=opp.employment_type,
                ctc_min_lpa=float(opp.ctc_min_lpa) if opp.ctc_min_lpa is not None else None,
                ctc_max_lpa=float(opp.ctc_max_lpa) if opp.ctc_max_lpa is not None else None,
                notice_period=opp.notice_period, recruiter_name=opp.recruiter_name,
                recruiter_email=opp.recruiter_email, recruiter_phone=opp.recruiter_phone,
                source_platform=opp.source_platform, jd_text=opp.jd_text,
            ) if opp else None,
            draft_type=draft.draft_type,
            draft_body=draft.body_text,
            quarantined=draft.quarantine_reason is not None,
            quarantine_reason=draft.quarantine_reason,
            rule_name=draft.rule_name,
            rule_reason=draft.rule_reason,
            fit_score=draft.fit_score,
            fit_rationale=draft.fit_rationale,
            fit_uncertain=draft.fit_uncertain,
            events=[
                NodeEventView(node=e.node, at=e.at, duration_ms=e.duration_ms, outcome=e.outcome)
                for e in events_raw
            ],
            created_at=draft.created_at,
        )


# =============================================================================
# POST /pending/{thread_id}/approve
# =============================================================================


@router.post("/{thread_id}/approve", response_model=ResolutionResponse)
def approve(
    thread_id: str, body: ApproveRequest, graph=Depends(get_graph)
) -> ResolutionResponse:
    """Approve the draft. Optional edited_body overrides the generated one.

    Per Phase 2 Q5: trust the human. If they edit the body, we do NOT re-run
    the outbound validator. The human is signing off on the exact bytes."""
    with session_scope() as s:
        _get_awaiting_message(s, thread_id)  # 404/409 guard

    # CONCEPT: Command(update=...) is LangGraph's way to inject state
    # into a paused thread and resume. The reducers apply as if a node
    # had returned this dict. `approval_status`, `approval_reason`,
    # `approved_body` all use REPLACE (default), so they just set the
    # value.
    config = {"configurable": {"thread_id": thread_id}}
    updates = {"approval_status": "approved", "approval_reason": body.note}
    if body.edited_body is not None:
        updates["approved_body"] = body.edited_body

    final_state = graph.invoke(Command(update=updates), config=config)

    return ResolutionResponse(
        thread_id=thread_id,
        final_status=final_state.get("final_status", MessageStatus.SENT_TO_GMAIL_DRAFTS),
        gmail_draft_id=final_state.get("gmail_draft_id"),
    )


# =============================================================================
# POST /pending/{thread_id}/reject
# =============================================================================


@router.post("/{thread_id}/reject", response_model=ResolutionResponse)
def reject(
    thread_id: str, body: RejectRequest, graph=Depends(get_graph)
) -> ResolutionResponse:
    """Reject the draft. Act node runs but skips Gmail; persist_final
    records rejected."""
    with session_scope() as s:
        _get_awaiting_message(s, thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(
        Command(update={"approval_status": "rejected", "approval_reason": body.reason}),
        config=config,
    )

    return ResolutionResponse(
        thread_id=thread_id,
        final_status=final_state.get("final_status", MessageStatus.REJECTED),
    )


# =============================================================================
# helpers
# =============================================================================


def _get_awaiting_message(session, thread_id: str) -> Message:
    msg = session.get(Message, thread_id)
    if msg is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"thread {thread_id} not found")
    if msg.status != MessageStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"thread {thread_id} is not awaiting approval (status={msg.status})",
        )
    return msg
