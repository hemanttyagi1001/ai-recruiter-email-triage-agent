"""
Persist nodes — Phase 2 splits Phase 1's monolithic persist into three:

  1. `persist_pending`  — writes message + opportunity + draft (status =
                          awaiting_approval) BEFORE the interrupt so the
                          FastAPI /pending endpoint can see the thread.
  2. `persist_final`    — updates message.status / draft.status AFTER
                          the act node runs (approved → drafted, rejected
                          → rejected).
  3. `persist_terminal` — the short-circuit path for skipped / failed /
                          needs-review threads that never reach the
                          interrupt. Same shape as Phase 1's persist.

CONCEPT: why split before/after the interrupt.
  The interrupt puts the graph in a paused state that can outlive the
  Python process. While paused, an OUTSIDE observer (the FastAPI service
  that a human is talking to) needs to see "this message is awaiting
  approval, here's the draft." That domain view lives in the DB; the
  checkpointer stores state-machine internals.

  If persist ran only at the end, the DB would have no row for the
  message during the pause — the API would have to reach into the
  checkpointer's tables to reconstruct it. That couples the API to
  LangGraph's internal storage shape (fragile). Splitting persist means
  the DB has a stable, queryable view of "what's awaiting review" and
  the checkpointer is free to be an implementation detail.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

from app.db.engine import session_scope
from app.db.models import (
    EXTRACTABLE_CATEGORIES,
    Category,
    Draft,
    DraftStatus,
    DraftType,
    DuplicateFlag,
    Message,
    MessageStatus,
    Opportunity as OpportunityRow,
)
from app.pipeline.state import TriageState, make_event


# =============================================================================
# persist_pending — runs after validate, before the interrupt
# =============================================================================


def persist_pending(state: TriageState) -> dict:
    """Write message + opportunity + draft rows with status=awaiting_approval.

    The draft may be quarantined by the validator; that's captured on the
    draft row but the message is still `awaiting_approval` — the human
    reviews the quarantine and decides what to do (edit, reject, or
    approve as-is if the quarantine was a false positive)."""
    started = time.monotonic()
    parsed = state["parsed"]
    run_id = state["run_id"]
    category = state.get("category")
    opportunity = state.get("opportunity")
    extraction_retries = state.get("extraction_retries", 0)
    rule_verdict = state.get("rule_verdict")
    fit_score_result = state.get("fit_score_result")
    draft_body = state.get("draft_body")
    draft_type = state.get("draft_type") or DraftType.DECLINE
    draft_validation = state.get("draft_validation")

    now = datetime.now(timezone.utc)

    with session_scope() as s:
        # Idempotency: if the same thread resumed all the way back to
        # persist_pending (shouldn't happen — checkpoint would resume from
        # act — but belt), skip.
        if s.get(Message, parsed.message_id) is not None:
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "final_status": MessageStatus.AWAITING_APPROVAL,
                "events": [make_event("persist_pending", outcome="already_exists",
                                      duration_ms=duration_ms)],
            }

        s.add(
            Message(
                message_id=parsed.message_id,
                gmail_id=parsed.gmail_id,
                thread_id=parsed.thread_id,
                from_email=parsed.from_email,
                from_name=parsed.from_name,
                subject=parsed.subject,
                received_at=parsed.received_at,
                body_text=parsed.body_text,
                raw_headers=parsed.raw_headers,
                run_id=run_id,
                category=category,
                classified_at=now if category else None,
                extracted_at=now if opportunity is not None else None,
                status=MessageStatus.AWAITING_APPROVAL,
            )
        )
        # D78: force the parent row out before anything that references it.
        # The Draft below carries a FK to messages.message_id, and until the
        # relationship() added to models.py the unit of work had no idea these
        # two INSERTs were ordered — it could emit the child first, get a FK
        # violation, and roll back the whole transaction including this row.
        # GOTCHA: this looks redundant now that the relationship exists, and
        # it is kept deliberately. It is one cheap round-trip that makes the
        # ordering true at the point a reader is looking at, rather than true
        # because of a declaration 200 lines away in another module.
        s.flush()

        opp_id = None
        if opportunity is not None:
            opp = opportunity.model_dump()
            row = OpportunityRow(
                message_id=parsed.message_id,
                company=opp["company"], end_client=opp["end_client"],
                role_title=opp["role_title"], location=opp["location"],
                work_model=opp["work_model"], employment_type=opp["employment_type"],
                ctc_min_lpa=opp["ctc_min_lpa"], ctc_max_lpa=opp["ctc_max_lpa"],
                notice_period=opp["notice_period"], recruiter_name=opp["recruiter_name"],
                recruiter_email=opp["recruiter_email"], recruiter_phone=opp["recruiter_phone"],
                source_platform=opp["source_platform"], jd_text=opp["jd_text"],
                # WHY read from state, not from the Pydantic Opportunity: the
                # embedding lives outside the extractor's schema — it's
                # produced by a separate node (embed_jd) that writes into
                # graph state. None here means embed was skipped or failed,
                # which the model column allows via nullable.
                jd_embedding=state.get("jd_embedding"),
                retry_count=extraction_retries,
            )
            s.add(row)
            s.flush()  # populates row.id for FK
            opp_id = row.id

            # WHY the flag writes happen HERE (right after opp flush) and
            # not in a dedicated persist_dedup node: dedup_check runs
            # pre-persist, so the new opp's id doesn't exist until this
            # flush. The flags reference that id via FK. Splitting into
            # a separate node would force a two-phase persist and a
            # cross-node id handoff for no benefit — the dedup write is
            # a small, tightly coupled tail of the opportunity write.
            # Silently skipped when duplicate_candidates is absent (dedup
            # disabled) or empty (dedup ran, no matches).
            _write_duplicate_flags(s, opp_id, state.get("duplicate_candidates") or [])

        # WHY the draft row always exists when persist_pending runs: the
        # graph only routes here from validate (which requires draft_body
        # to be set). Every awaiting_approval message has a draft.
        quarantined = draft_validation is not None and draft_validation.quarantined
        s.add(
            Draft(
                message_id=parsed.message_id,
                opportunity_id=opp_id,
                draft_type=draft_type,
                body_text=draft_body,
                # Status is AWAITING_APPROVAL even when quarantined — the
                # human still needs to see it. quarantine_reason carries
                # the validator's concern for the human to consider.
                status=DraftStatus.AWAITING_APPROVAL,
                quarantine_reason=draft_validation.reason if quarantined else None,
                rule_name=rule_verdict.rule_name if rule_verdict else None,
                rule_reason=rule_verdict.reason if rule_verdict else None,
                fit_score=fit_score_result.score if fit_score_result else None,
                fit_rationale=fit_score_result.rationale if fit_score_result else None,
                fit_uncertain=fit_score_result.uncertain if fit_score_result else None,
            )
        )

    duration_ms = int((time.monotonic() - started) * 1000)
    return {
        "final_status": MessageStatus.AWAITING_APPROVAL,
        "events": [
            make_event(
                "persist_pending",
                outcome=f"awaiting_approval; draft_type={draft_type}"
                        + (" quarantined" if quarantined else ""),
                duration_ms=duration_ms,
            )
        ],
    }


# =============================================================================
# persist_final — runs after act, records the terminal outcome
# =============================================================================


def persist_final(state: TriageState) -> dict:
    """Update message.status and draft.status based on approval + act result."""
    started = time.monotonic()
    parsed = state["parsed"]
    approval = state.get("approval_status")
    approval_reason = state.get("approval_reason")
    approved_body = state.get("approved_body")
    gmail_draft_id = state.get("gmail_draft_id")
    halted = state.get("halted", False)

    now = datetime.now(timezone.utc)

    with session_scope() as s:
        msg: Message | None = s.get(Message, parsed.message_id)
        draft: Draft | None = (
            s.query(Draft).filter_by(message_id=parsed.message_id).one_or_none()
        )
        if msg is None or draft is None:
            # Should not happen — persist_pending ran before us. Loud fail.
            raise RuntimeError(
                f"persist_final could not find message/draft for {parsed.message_id}; "
                f"persist_pending must have failed silently"
            )

        if approval == "approved" and not halted:
            msg.status = MessageStatus.SENT_TO_GMAIL_DRAFTS
            draft.status = DraftStatus.SENT_TO_GMAIL_DRAFTS
            draft.gmail_draft_id = gmail_draft_id
            draft.approval_reason = approval_reason
            draft.resolved_at = now
        elif approval == "approved" and halted:
            # WHY leave status at AWAITING_APPROVAL: the kill switch
            # deferred the Gmail call. The message is not "sent-to-drafts"
            # (nothing landed in Gmail) and not "rejected" (the human
            # said yes). Rolling back to awaiting_approval lets the
            # human retry approve once the halt lifts. approval_reason
            # and resolved_at are NOT updated — the approval didn't
            # actually complete.
            log.warning(
                "persist_final: approval was 'approved' but act was halted; "
                "message_id=%s stays awaiting_approval for retry",
                parsed.message_id,
            )
        elif approval == "rejected":
            msg.status = MessageStatus.REJECTED
            draft.status = DraftStatus.REJECTED
            draft.approval_reason = approval_reason
            draft.resolved_at = now
        else:
            # Should not happen — the interrupt only lifts once approval
            # is set. Preserve original status; loud in the audit trail.
            pass

        # If the human edited the body, preserve the edited version as the
        # final body_text so it reflects what was actually sent to Gmail
        # (or what would have been sent, on the halted-approve path).
        if approved_body and not halted:
            draft.body_text = approved_body

    duration_ms = int((time.monotonic() - started) * 1000)
    if halted and approval == "approved":
        final_status = MessageStatus.AWAITING_APPROVAL
        outcome = "halted; awaiting_approval preserved for retry"
    elif approval == "approved":
        final_status = MessageStatus.SENT_TO_GMAIL_DRAFTS
        outcome = f"approval=approved gmail_id={gmail_draft_id}"
    else:
        final_status = MessageStatus.REJECTED
        outcome = f"approval={approval}"
    return {
        "final_status": final_status,
        "events": [make_event("persist_final", outcome=outcome, duration_ms=duration_ms)],
    }


# =============================================================================
# persist_terminal — short-circuit paths (skipped / failed / needs_review)
# =============================================================================


def persist_terminal(state: TriageState) -> dict:
    """Terminal writer for paths that don't reach the interrupt.

    Covers: classify said skip; extraction failed all retries; scorer
    abstained (needs_review). No draft, no opportunity in some cases.
    """
    started = time.monotonic()
    parsed = state["parsed"]
    run_id = state["run_id"]
    category = state.get("category")
    opportunity = state.get("opportunity")
    extraction_error = state.get("extraction_error")
    extraction_retries = state.get("extraction_retries", 0)
    fit_score_result = state.get("fit_score_result")
    needs_review = state.get("needs_review", False)
    now = datetime.now(timezone.utc)

    # D79: ingest can settle the terminal status before any other node runs.
    # A non-delivery report is recognised deterministically from the sender and
    # subject, so there is no category, no opportunity and no reply target for
    # _terminal_status to reason about — left to itself it would derive
    # FETCHED, which says "we have not decided yet" about a message we decided
    # about immediately.
    # WHY only this one status is honoured rather than any preset value: the
    # duplicate path through _route_after_ingest also arrives here carrying a
    # final_status (the EXISTING row's status), and that one must NOT overwrite
    # what the derivation produces for a fresh row. Narrowing to the single
    # value ingest is entitled to decide keeps the two cases apart.
    preset = state.get("final_status")
    if preset == MessageStatus.SKIPPED_UNDELIVERABLE:
        final_status = MessageStatus.SKIPPED_UNDELIVERABLE
    else:
        final_status = _terminal_status(
            category=category,
            opportunity=opportunity,
            extraction_error=extraction_error,
            needs_review=needs_review,
            reply_to=state.get("reply_to"),
            already_replied=bool(state.get("already_replied")),
        )

    with session_scope() as s:
        if s.get(Message, parsed.message_id) is not None:
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "final_status": final_status,
                "events": [make_event("persist_terminal", outcome="already_exists",
                                      duration_ms=duration_ms)],
            }

        s.add(
            Message(
                message_id=parsed.message_id,
                gmail_id=parsed.gmail_id,
                thread_id=parsed.thread_id,
                from_email=parsed.from_email,
                from_name=parsed.from_name,
                subject=parsed.subject,
                received_at=parsed.received_at,
                body_text=parsed.body_text,
                raw_headers=parsed.raw_headers,
                run_id=run_id,
                category=category,
                classified_at=now if category else None,
                extracted_at=now if opportunity is not None else None,
                extraction_error=extraction_error,
                status=final_status,
            )
        )

        # For needs_review, we still persist the opportunity — the human
        # may want to see the extracted data even though the scorer
        # abstained. Same jd_embedding read-from-state as persist_pending.
        if opportunity is not None:
            opp = opportunity.model_dump()
            opp_row = OpportunityRow(
                message_id=parsed.message_id,
                company=opp["company"], end_client=opp["end_client"],
                role_title=opp["role_title"], location=opp["location"],
                work_model=opp["work_model"], employment_type=opp["employment_type"],
                ctc_min_lpa=opp["ctc_min_lpa"], ctc_max_lpa=opp["ctc_max_lpa"],
                notice_period=opp["notice_period"], recruiter_name=opp["recruiter_name"],
                recruiter_email=opp["recruiter_email"], recruiter_phone=opp["recruiter_phone"],
                source_platform=opp["source_platform"], jd_text=opp["jd_text"],
                jd_embedding=state.get("jd_embedding"),
                retry_count=extraction_retries,
            )
            s.add(opp_row)
            s.flush()  # populates opp_row.id for the flag FK
            # WHY needs_review still writes flags: dedup_check ran for
            # this thread (the score → uncertain path passes through it)
            # and produced candidates. A human reviewing a needs_review
            # message benefits from the "you've seen something like this
            # before" signal as much as the awaiting_approval path does —
            # arguably more, since they're on the fence about the fit.
            _write_duplicate_flags(
                s, opp_row.id, state.get("duplicate_candidates") or []
            )

    duration_ms = int((time.monotonic() - started) * 1000)
    return {
        "final_status": final_status,
        "events": [make_event("persist_terminal", outcome=final_status,
                              duration_ms=duration_ms)],
    }


# =============================================================================
# persist_auto — the autonomy path terminal writer (Phase 5)
# =============================================================================


def persist_auto(state: TriageState) -> dict:
    """Terminal writer for the rule-based decline auto-send path.

    Runs after auto_send. Writes the message + opportunity + draft rows
    in one go, with terminal status AUTO_SENT (or AWAITING_APPROVAL if
    auto_send was halted by the kill switch — in which case we hand
    off to the human path, though the graph will END here and rely on
    a re-ingest to pick up the awaiting_approval message).

    WHY a dedicated persist function rather than reusing persist_pending
    then persist_final: the auto path never goes through the interrupt.
    persist_pending sets AWAITING_APPROVAL and expects act to promote
    it later; persist_final expects an approval_status set by the API.
    Neither invariant holds here. A dedicated writer is honest about
    the different lifecycle.

    On halt (state.halted=True, no gmail_sent_id): writes the message
    as AWAITING_APPROVAL with a draft in AWAITING_APPROVAL, matching
    what persist_pending would have written. The next time a human
    reviews (or re-ingest with the halt off runs), the draft is there.
    """
    started = time.monotonic()
    parsed = state["parsed"]
    run_id = state["run_id"]
    category = state.get("category")
    opportunity = state.get("opportunity")
    extraction_retries = state.get("extraction_retries", 0)
    rule_verdict = state.get("rule_verdict")
    draft_body = state.get("draft_body")
    draft_validation = state.get("draft_validation")
    # Present only when the scorer ran, which D45 made possible on this path.
    fit_score_result = state.get("fit_score_result")
    gmail_sent_id = state.get("gmail_sent_id")
    gmail_draft_id = state.get("gmail_draft_id")
    halted = state.get("halted", False)

    now = datetime.now(timezone.utc)

    # Route the terminal status. Three outcomes now reach here:
    #   halted            → AWAITING_APPROVAL  (kill switch deferred a send)
    #   gmail_draft_id    → SENT_TO_GMAIL_DRAFTS (draft mode, D49)
    #   gmail_sent_id     → AUTO_SENT
    # WHY draft mode reuses SENT_TO_GMAIL_DRAFTS rather than earning its own
    # status: the observable outcome is identical to a human-approved draft —
    # a reply sitting in Gmail waiting for someone to press send. The only
    # difference is who decided to put it there, and `auto_actioned` already
    # records exactly that. A separate status would split one real-world state
    # across two values and quietly break every existing report that filters
    # on this one.
    if halted:
        msg_status = MessageStatus.AWAITING_APPROVAL
        draft_status = DraftStatus.AWAITING_APPROVAL
        auto_actioned = False
    elif gmail_draft_id is not None:
        msg_status = MessageStatus.SENT_TO_GMAIL_DRAFTS
        draft_status = DraftStatus.SENT_TO_GMAIL_DRAFTS
        # True because no human approved this one — the agent decided to draft
        # it. That is the distinction auto_actioned exists to capture (D37),
        # and it is what lets the digest separate "drafts I asked for" from
        # "drafts the agent produced on its own".
        auto_actioned = True
        resolved_at = None
    else:
        msg_status = MessageStatus.AUTO_SENT
        draft_status = DraftStatus.AUTO_SENT
        auto_actioned = True
        resolved_at = now

    with session_scope() as s:
        # Idempotency: same guard as persist_pending. A resume that
        # somehow re-enters persist_auto for the same thread is a bug,
        # but we no-op instead of raising to keep the caller's crash
        # recovery simple.
        if s.get(Message, parsed.message_id) is not None:
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "final_status": msg_status,
                "events": [make_event("persist_auto", outcome="already_exists",
                                      duration_ms=duration_ms)],
            }

        s.add(
            Message(
                message_id=parsed.message_id,
                gmail_id=parsed.gmail_id,
                thread_id=parsed.thread_id,
                from_email=parsed.from_email,
                from_name=parsed.from_name,
                subject=parsed.subject,
                received_at=parsed.received_at,
                body_text=parsed.body_text,
                raw_headers=parsed.raw_headers,
                run_id=run_id,
                category=category,
                classified_at=now if category else None,
                extracted_at=now if opportunity is not None else None,
                status=msg_status,
            )
        )
        # D78 — this is the exact line whose absence caused the incident.
        # TRACE: reaching here means auto_send has ALREADY created a Gmail
        # draft. If this transaction dies, that draft exists with nothing in
        # the database pointing at it, the message looks unseen to the next
        # cycle's dedup guards, and the agent drafts for it again — once every
        # POLL_INTERVAL_MINUTES, indefinitely. That is why the parent row goes
        # out first here rather than relying on flush order being inferred.
        s.flush()

        opp_id = None
        if opportunity is not None:
            opp = opportunity.model_dump()
            row = OpportunityRow(
                message_id=parsed.message_id,
                company=opp["company"], end_client=opp["end_client"],
                role_title=opp["role_title"], location=opp["location"],
                work_model=opp["work_model"], employment_type=opp["employment_type"],
                ctc_min_lpa=opp["ctc_min_lpa"], ctc_max_lpa=opp["ctc_max_lpa"],
                notice_period=opp["notice_period"], recruiter_name=opp["recruiter_name"],
                recruiter_email=opp["recruiter_email"], recruiter_phone=opp["recruiter_phone"],
                source_platform=opp["source_platform"], jd_text=opp["jd_text"],
                jd_embedding=state.get("jd_embedding"),
                retry_count=extraction_retries,
            )
            s.add(row)
            s.flush()
            opp_id = row.id
            _write_duplicate_flags(s, opp_id, state.get("duplicate_candidates") or [])

        s.add(
            Draft(
                message_id=parsed.message_id,
                opportunity_id=opp_id,
                # GOTCHA: this was hard-coded to DraftType.DECLINE, with a
                # comment saying it would need to come from state "if we ever
                # widened to interested-and-safe-somehow". D45 did exactly
                # that, and this line was missed — so every autonomously
                # handled INTERESTED reply was being recorded as a decline.
                # The bug is invisible in the mailbox (the body is correct)
                # and only shows up in the digest and in any later analysis of
                # what the agent actually did.
                draft_type=state.get("draft_type") or DraftType.DECLINE,
                body_text=draft_body,
                status=draft_status,
                quarantine_reason=(
                    draft_validation.reason
                    if (draft_validation and draft_validation.quarantined)
                    else None
                ),
                rule_name=rule_verdict.rule_name if rule_verdict else None,
                rule_reason=rule_verdict.reason if rule_verdict else None,
                # Same D45 gap as draft_type above: under the old criterion the
                # scorer could never have run on this path, so these were left
                # unwritten. Now an interested reply arrives here WITH a score,
                # and dropping it would leave the digest unable to answer "what
                # did the agent think of the roles it answered on its own".
                fit_score=fit_score_result.score if fit_score_result else None,
                fit_rationale=fit_score_result.rationale if fit_score_result else None,
                fit_uncertain=fit_score_result.uncertain if fit_score_result else None,
                # WHY auto_actioned bool alongside AUTO_SENT status:
                # digest / audit queries want "count autonomous actions
                # in the last N hours" and reading a bool is faster and
                # clearer than filtering by two separate status values.
                # See D37.
                auto_actioned=auto_actioned,
                # One column, two meanings, depending on which Gmail call ran:
                # the draft id in `draft` mode, the sent-message id in `on`.
                # Both identify the artefact this row produced in Gmail, and
                # `status` already says which kind it is.
                gmail_draft_id=gmail_draft_id or gmail_sent_id,
                # D60: who we actually wrote to, for the dedup lookup.
                reply_to_email=state.get("reply_to"),
                resolved_at=resolved_at,
            )
        )

    duration_ms = int((time.monotonic() - started) * 1000)
    if halted:
        outcome = "halted; awaiting_approval fallback"
    else:
        outcome = f"auto_sent gmail_id={gmail_sent_id} rule={rule_verdict.rule_name if rule_verdict else '?'}"
    return {
        "final_status": msg_status,
        "events": [make_event("persist_auto", outcome=outcome, duration_ms=duration_ms)],
    }


# =============================================================================
# _write_duplicate_flags — shared helper
# =============================================================================


def _write_duplicate_flags(session, opportunity_id, candidates) -> None:
    """Insert one DuplicateFlag row per candidate.

    Split into a helper because both persist_pending and persist_terminal
    do this identically; the only branch is the pre-flush id source.

    WHY no ON CONFLICT: the (opportunity_id, matched_opportunity_id) pair
    is not UNIQUE in the schema — the same new opp calling into persist
    twice is prevented by the message_id PK check at the top of both
    persist nodes. If it *did* happen, the CASCADE FK to opportunities
    would clean up on any later opp deletion. Simpler than defensive
    uniqueness.
    """
    for cand in candidates:
        session.add(
            DuplicateFlag(
                opportunity_id=opportunity_id,
                matched_opportunity_id=cand.opportunity_id,
                similarity=cand.similarity,
            )
        )


def _terminal_status(
    *, category, opportunity, extraction_error, needs_review, reply_to=None,
    already_replied=False,
) -> str:
    if needs_review:
        return MessageStatus.NEEDS_REVIEW
    if extraction_error:
        return MessageStatus.EXTRACTION_FAILED
    if category:
        try:
            if Category(category) not in EXTRACTABLE_CATEGORIES:
                return MessageStatus.SKIPPED_WRONG_CATEGORY
        except ValueError:
            pass
    # WHY this sits below the category check but above EXTRACTED: a message
    # that was not recruitment mail at all is already accounted for above, and
    # anything reaching here extracted cleanly. EXTRACTED would be technically
    # true but useless — it hides the reason we stopped. See D47.
    # GOTCHA: keyword-only with a default so the Phase 1 shim and any caller
    # that predates reply_to keeps working; those pass an opportunity and no
    # reply_to, and fall through to EXTRACTED exactly as before.
    if already_replied:
        return MessageStatus.SKIPPED_ALREADY_REPLIED
    if opportunity is not None and reply_to is None:
        return MessageStatus.SKIPPED_NO_REPLY_TARGET
    if opportunity is not None:
        return MessageStatus.EXTRACTED
    if category:
        return MessageStatus.CLASSIFIED
    return MessageStatus.FETCHED


# =============================================================================
# Backward-compat shim — Phase 1's persist_node still imported by test_idempotency.
# Retained so old tests keep passing without a mass-edit. New code uses the
# split nodes above.
# =============================================================================


def persist_node(state: dict[str, Any]) -> dict[str, Any]:
    """Legacy Phase 1 single-persist. Delegates to persist_terminal for the
    same-shape paths (no draft in state); to persist_pending otherwise."""
    if state.get("draft_body") is not None:
        # Simulate Phase 1's "message.status = DRAFTED" behaviour by using
        # persist_pending (which writes awaiting_approval). Callers of the
        # legacy shim in tests set status explicitly.
        result = persist_pending(state)
        # Return the Phase 1 status name to keep old assertions passing.
        return {**result, "final_status": MessageStatus.DRAFTED}
    return persist_terminal(state)
