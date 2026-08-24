"""
Act node — the one Gmail-mutating step in the pipeline.

CONCEPT: what "act" means in a human-in-the-loop graph.
  In Phase 1, the pipeline had no interrupt: it wrote a "pending review"
  draft to the DB and stopped. A human read the DB and manually pasted the
  draft into their mail client.

  Phase 2 makes the last-mile visible in the graph itself:
    validate → persist_pending → INTERRUPT → act → persist_final
                                 ^^^^^^^^^          ^^^^^^^^^^^^^
                                 human decides      records outcome

  When the graph resumes past the interrupt, `state["approval_status"]`
  has been set by the API endpoint. If approved, act calls Gmail's
  drafts.create. If rejected, act is a no-op (persist_final records the
  rejection). Either way, the ACT step runs — that's why it exists as a
  named node instead of the API doing the Gmail call itself. Keeping the
  action inside the graph means it appears in the audit trail (events
  list), gets the same crash-recovery guarantees as other nodes, and
  survives process restart between "user clicked approve" and "Gmail
  API responded."

  If act crashes mid-call (network blip), the checkpoint state before act
  is still on disk; a rerun resumes and retries. Idempotency of retry is
  best-effort — Gmail may end up with a duplicate draft — but a duplicate
  DRAFT (which the user reviews before sending) is a much smaller failure
  than a duplicate SEND.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from app.config import settings
from app.gmail.client import GmailClient
from app.kill_switch import is_send_halted
from app.pipeline.state import TriageState, make_event

log = logging.getLogger(__name__)


def _resume_attachment(state) -> "Path | None":
    """The resume path, when the file is actually there.

    WHY the existence check is here rather than at startup: the file is a bind
    mount, so it can disappear between boot and send. Returning None with a
    warning means the reply still goes out — a missing CV must never be the
    reason a recruiter hears nothing back.
    """
    # D61 restores the request gate that D57 briefly removed. An unsolicited
    # CV to every stranger is a lot of outbound PDF, and a recruiter who wants
    # one asks — `app/rules/resume_request.py` matched 19/19 real phrasings.
    # GOTCHA: this and the draft wording must agree. If this returns None the
    # body must not claim a CV is attached, which is why `resume_requested`
    # feeds both.
    if not state.get("resume_requested"):
        return None
    path = settings.resume_path
    if path is None:
        log.warning("RESUME_PATH is not configured; sending without a CV attached")
        return None
    if not path.exists():
        log.warning("resume %s does not exist; sending without a CV attached", path)
        return None
    return path


def make_act_node(gmail: GmailClient):
    """Factory — the Gmail client is injected so tests can substitute a fake."""

    def act_node(state: TriageState) -> dict:
        started = time.monotonic()
        parsed = state["parsed"]
        approval = state.get("approval_status")

        if approval != "approved":
            # Rejected path — do not touch Gmail. persist_final marks the
            # draft rejected and records approval_reason.
            reason = state.get("approval_reason", "(no reason given)")
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "events": [
                    make_event("act", outcome=f"rejected: {reason[:80]}", duration_ms=duration_ms)
                ]
            }

        # Approved path — write to Gmail Drafts.
        # WHY approved_body || draft_body: the approve endpoint sets
        # approved_body only when the human edited the draft; otherwise
        # we use the machine-generated draft_body as-is.
        body = state.get("approved_body") or state["draft_body"]
        subject = _reply_subject(parsed.subject)

        # KILL SWITCH — checked right before the Gmail call, per D36.
        # A halted state means the human-approved send is deferred:
        # draft is NOT created in Gmail; message stays awaiting_approval
        # in the DB (already persisted by persist_pending); the human
        # can retry the approve once the halt is lifted.
        if is_send_halted():
            duration_ms = int((time.monotonic() - started) * 1000)
            log.warning(
                "act halted by kill switch for message_id=%s; draft NOT created",
                parsed.message_id,
            )
            return {
                "halted": True,
                "events": [
                    make_event(
                        "act", outcome="halted by kill switch",
                        duration_ms=duration_ms,
                    )
                ],
            }

        # TRACE: single Gmail API call — users().drafts().create.
        # See GmailClient.create_draft docstring for header details.
        # GOTCHA: the recipient is the RESOLVED reply target, not
        # parsed.from_email. Portal-relayed mail arrives from an opaque
        # forwarding address while the recruiter's real mailbox sits in the
        # body; reply_target.resolve_reply_target picks the latter. The `or`
        # fallback covers threads checkpointed before reply_to existed —
        # those resume with the field absent and keep the old behaviour
        # rather than crashing on a missing key. See D47.
        gmail_draft_id = gmail.create_draft(
            to=state.get("reply_to") or parsed.from_email,
            subject=subject,
            body_text=body,
            in_reply_to_message_id=parsed.message_id,
            gmail_thread_id=parsed.thread_id,
            attachment_path=_resume_attachment(state),
        )
        log.info("Created Gmail draft %s for message %s", gmail_draft_id, parsed.message_id)
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "gmail_draft_id": gmail_draft_id,
            "events": [
                make_event("act", outcome=f"drafted gmail_id={gmail_draft_id}",
                           duration_ms=duration_ms)
            ],
        }

    return act_node


def _reply_subject(original: str) -> str:
    """RFC 5322 reply-prefix convention: prepend "Re: " unless it's already there."""
    stripped = (original or "(no subject)").strip()
    if stripped.lower().startswith("re:"):
        return stripped
    return f"Re: {stripped}"
