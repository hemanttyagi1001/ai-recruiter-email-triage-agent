"""
Auto-send node — Phase 5.

The one graph node that autonomously sends a reply to a recruiter. Only
reached when `_route_after_validate` decides the draft meets the
autonomy criterion (rule-fired decline, validator passed, no fit score,
no duplicate flags). See D33 for the criterion and its blast-radius ×
reversibility rationale.

CONCEPT: what "autonomous" means here, precisely.
  The graph does not pause for a human. It calls Gmail's messages.send
  with a template-generated body. The recruiter sees the reply within
  seconds. There is no undo — Gmail has no unsend after send is
  confirmed. Every guarantee this path relies on is upstream:

    - draft_type=DECLINE + rule_verdict != None: means the send
      decision came from a Python function, not an LLM.
    - draft body: template rendering only (D12). No LLM in the
      drafting path means no hallucinated claim in the sent text.
    - validator: PII scan and length cap ran and did not quarantine.
    - kill switch: consulted right here, right before the API call.

  Any single one of these disqualifies the auto path. Together they
  define a narrow window where "send without asking" is defensible.

CONCEPT: kill-switch check location matters.
  is_send_halted() is called INSIDE this function, in the two lines
  before gmail.send_reply. Not at module load, not at graph compile,
  not at node factory time. This means `psql -c "UPDATE system_flags
  SET value=true WHERE key='sends_halted'"` takes effect on the very
  next message through this node — no restart, no wait. See D36.
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


def make_auto_send_node(gmail: GmailClient):
    """Factory — gmail is injected so tests substitute a fake."""

    def auto_send_node(state: TriageState) -> dict:
        started = time.monotonic()
        parsed = state["parsed"]
        body = state["draft_body"]
        subject = _reply_subject(parsed.subject)
        # GOTCHA: send to the RESOLVED target, never parsed.from_email. For a
        # portal-relayed message those differ — from_email is an opaque
        # forwarding address like
        # `ashabWlkd2VzdGNvbnN1bHRhbnRzLm5ldA==@naukri.com` while
        # reply_to is the recruiter's real mailbox recovered from the body.
        # The fallback is defensive only: routing guarantees reply_to is set
        # before anything reaches this node (see _route_after_extract), and a
        # message with neither cannot arrive here at all.
        recipient = state.get("reply_to") or parsed.from_email

        # DRAFT MODE — checked BEFORE the kill switch, and that ordering is the
        # whole decision.
        # CONCEPT: the kill switch means "no mail leaves this mailbox". A draft
        #   does not leave the mailbox — it sits in Drafts until a human presses
        #   send. So drafting is permitted while halted, and the operator can
        #   leave the switch armed for the entire soak period without it
        #   blocking anything they actually want to happen.
        # WHY that is safe rather than a loophole: in this mode the send call
        #   below is unreachable. There is no configuration where `draft`
        #   dispatches mail. The switch regains its full meaning the moment the
        #   mode becomes `on`, which is a deliberate env change plus a restart.
        # GOTCHA: this leaves auto_send asymmetric with act, which DOES consult
        #   the kill switch before its own create_draft. The asymmetry is
        #   intentional and worth knowing when reading the two side by side —
        #   act runs only after a human approved a specific message, so an
        #   operator halting mid-review means "stop everything you are doing",
        #   whereas here drafting IS the requested steady state. See D49.
        if settings.auto_send_mode == "draft":
            gmail_draft_id = gmail.create_draft(
                to=recipient,
                subject=subject,
                body_text=body,
                in_reply_to_message_id=parsed.message_id,
                gmail_thread_id=parsed.thread_id,
                attachment_path=_resume_attachment(state),
            )
            duration_ms = int((time.monotonic() - started) * 1000)
            log.info(
                "auto_draft created gmail_draft_id=%s for message_id=%s to %s",
                gmail_draft_id, parsed.message_id, recipient,
            )
            return {
                "gmail_draft_id": gmail_draft_id,
                "halted": False,
                "events": [
                    make_event(
                        "auto_send",
                        outcome=f"drafted gmail_draft_id={gmail_draft_id}",
                        duration_ms=duration_ms,
                    )
                ],
            }

        # D58 — the D56 prohibition on sending LLM-written bodies was REMOVED
        # here, at the operator's direction, restated after the risk was set
        # out in full. What used to sit at this point diverted
        # `draft_source == "llm"` to create_draft regardless of mode.
        #
        # What that guarantee was, so a future reader knows what is gone:
        #   To answer a recruiter's question the drafting prompt must contain
        #   that recruiter's untrusted words. Fencing makes an injected
        #   instruction produce TEXT rather than an action (D13 still holds —
        #   the LLM surface is tool-free). D56's argument was that such text is
        #   harmless only because a human reads it before it goes anywhere.
        #   With this removed, model-written text shaped by a stranger's email
        #   can now reach a recruiter's inbox unread, under the user's name.
        #
        # WHAT STILL STANDS, and it is not nothing:
        #   - D14 quarantine. A body tripping the PAN/Aadhaar scan or the
        #     length cap is still routed to a human, in every mode. Config
        #     cannot open that gate.
        #   - D36 kill switch, checked two lines below. It is now the ONLY
        #     runtime control between the model and the recruiter, which makes
        #     it considerably more load-bearing than it was yesterday.
        #   - The system prompt's hard rules: never state an unlisted fact,
        #     never name a figure below the expectation.
        # KILL SWITCH — read fresh, no cache. See module docstring.
        if is_send_halted():
            duration_ms = int((time.monotonic() - started) * 1000)
            log.warning(
                "auto_send halted by kill switch for message_id=%s; "
                "routing to human-approval path",
                parsed.message_id,
            )
            # Setting halted=True routes to persist_pending downstream,
            # so the message becomes awaiting_approval and a human sees
            # it once the halt lifts. No Gmail call was made.
            return {
                "halted": True,
                "events": [
                    make_event(
                        "auto_send",
                        outcome="halted by kill switch",
                        duration_ms=duration_ms,
                    )
                ],
            }

        # DRY RUN — everything above this point has already happened for real:
        # classify, extract, rules, scoring, drafting, validation, and the kill
        # switch check. Only the irreversible step is skipped.
        # WHY return halted=True: it reuses the existing "did not send" edge
        # rather than inventing a third outcome. persist_auto already knows to
        # leave such a message awaiting_approval, which is exactly what a dry
        # run wants — the draft is queued at /pending so the operator can read
        # what WOULD have gone out and compare it against their own judgement.
        # GOTCHA: the test is `!= "on"`, not `== "dry_run"`. Checking for
        # dry_run would let mode="off" fall through and SEND — the exact
        # inverse of what "off" means. Routing should never deliver an "off"
        # thread here, but a node whose safe behaviour depends on the router
        # having done its job is one refactor away from mailing strangers.
        # Send is opt-in by exact match; everything else declines to act.
        if settings.auto_send_mode != "on":
            duration_ms = int((time.monotonic() - started) * 1000)
            log.warning(
                "NOT SENDING (auto_send_mode=%s) — would have sent to %s | "
                "subject=%r | %d chars. Set AUTO_SEND_MODE=on to send for "
                "real. Body:\n%s",
                settings.auto_send_mode, recipient, subject,
                len(body), body,
            )
            return {
                "halted": True,
                "events": [
                    make_event(
                        "auto_send",
                        outcome=(
                            f"{settings.auto_send_mode}: would send to "
                            f"{recipient}"
                        ),
                        duration_ms=duration_ms,
                    )
                ],
            }

        # TRACE: single POST to users().messages().send with retry
        # DISABLED (see GmailClient.send_reply for why). A failure here
        # raises PermanentExternalError, which the ingest CLI catches
        # and turns into a dead_letters row. No partial state is left
        # behind because no persist has run yet for this message.
        gmail_sent_id = gmail.send_reply(
            to=recipient,
            subject=subject,
            body_text=body,
            in_reply_to_message_id=parsed.message_id,
            gmail_thread_id=parsed.thread_id,
            attachment_path=_resume_attachment(state),
        )
        log.info(
            "auto_send sent gmail_id=%s for message_id=%s (rule=%s)",
            gmail_sent_id, parsed.message_id,
            state.get("rule_verdict").rule_name if state.get("rule_verdict") else "?",
        )
        # D68: mark the INBOUND message read, now that it has been answered.
        # TRACE: ordered strictly after the send, and only on this path. The
        # dry_run and draft branches above have already returned, so nothing
        # is marked read unless a real email actually left the building —
        # "read" in the inbox means "answered", not "looked at".
        # GOTCHA: parsed.gmail_id, NOT parsed.message_id. modify() addresses
        # messages by Gmail's internal id; the RFC 5322 Message-ID is our
        # primary key (D2) and Gmail's API does not accept it.
        # WHY a second guard when GmailClient.mark_read already swallows
        # everything: that promise lives in another module, and this call site
        # is the one that must not fail. A fake in a test, a future refactor,
        # or a client swapped for another provider need only raise once to
        # dead-letter a message whose reply was already delivered — and the
        # next ingest would then send a duplicate. The cost of the redundancy
        # is four lines; the cost of trusting the contract is a second email
        # to a recruiter.
        try:
            marked = gmail.mark_read(parsed.gmail_id)
        except Exception:
            log.exception(
                "mark_read raised for gmail_id=%s after a SUCCESSFUL send; "
                "swallowing so the delivered reply is not dead-lettered",
                parsed.gmail_id,
            )
            marked = False
        if not marked:
            log.info(
                "reply sent for message_id=%s but it could not be marked read; "
                "it stays unread in the inbox", parsed.message_id,
            )
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "gmail_sent_id": gmail_sent_id,
            "halted": False,
            "events": [
                make_event(
                    "auto_send",
                    outcome=(
                        f"sent gmail_id={gmail_sent_id}"
                        + ("" if marked else "; mark_read failed")
                    ),
                    duration_ms=duration_ms,
                )
            ],
        }

    return auto_send_node


def _reply_subject(original: str) -> str:
    """Same RFC 5322 reply-prefix convention as act._reply_subject.

    Duplicated instead of imported to keep this module standalone —
    act.py may evolve in ways we don't want to drag into the autonomy
    path (e.g., extra formatting for human-visible drafts).
    """
    stripped = (original or "(no subject)").strip()
    if stripped.lower().startswith("re:"):
        return stripped
    return f"Re: {stripped}"
