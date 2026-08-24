"""
Have we already written to this recruiter?

What this module does and why it exists here: a recruiter who mails three
times about three roles is still one person, and three near-identical replies
from an agent is worse than one. This answers "have we sent this address a
reply before" with a single indexed lookup, so a repeat message terminates
before it costs a fit score or a drafting call.

CONCEPT: dedup on the RESOLVED address, never on the sender.
  `messages.from_email` is frequently not who we wrote to — D47 prefers the
  recruiter address the extractor found in the body, and D59 base64-decodes
  Naukri relay headers. The same human can therefore arrive as:
      asha@midwestconsultants.net              (direct)
      ashabWlkd2VzdGNvbnN1bHRhbnRzLm5ldA==@naukri.com   (relay)
  Two different senders, one mailbox. Deduping on `from_email` would treat
  them as strangers and reply twice; deduping on the resolved target sees one
  person. This is exactly why migration 0006 records `drafts.reply_to_email`
  rather than deriving it at query time.

CONCEPT: sent-only, by operator choice (D60).
  A draft that was created but never sent does NOT count as a reply. The
  recruiter has heard nothing, so they should still get one. The cost of that
  choice: if drafts pile up unsent in Gmail, the agent will draft for the same
  person again. Under AUTO_SEND_MODE=on that is moot — everything either sends
  or fails loudly.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select

from app.db.engine import session_scope
from app.db.models import Draft, DraftStatus

log = logging.getLogger(__name__)

# Statuses that mean a real email reached the recruiter. AWAITING_APPROVAL and
# SENT_TO_GMAIL_DRAFTS are deliberately absent: a draft sitting in the Drafts
# folder was never delivered, and the recruiter is still waiting.
_DELIVERED: frozenset[str] = frozenset({DraftStatus.AUTO_SENT})


def already_replied(reply_to: str | None) -> bool:
    """True if a reply has already been SENT to this address.

    TRACE: one indexed SELECT per message, run immediately after the reply
    target is resolved and before embed_jd. A hit short-circuits the rest of
    the pipeline, so the marginal cost of a recruiter's second, third and
    fourth email is classify + extract only — no embedding, no fit score, no
    drafting call.

    GOTCHA: fails OPEN. A database error here returns False, meaning "not
    replied", meaning we reply. The alternative — failing closed — would make
    a transient DB blip look identical to "already handled" and silently drop
    a real recruiter's first contact. A duplicate reply is embarrassing; a
    dropped one is an opportunity lost with no trace.
    """
    if not reply_to:
        return False
    try:
        with session_scope() as s:
            # lower() on both sides to match the functional index from
            # migration 0006 — email domains are case-insensitive and
            # recruiters sign off inconsistently.
            hit = s.execute(
                select(Draft.id)
                .where(func.lower(Draft.reply_to_email) == reply_to.strip().lower())
                .where(Draft.status.in_(tuple(_DELIVERED)))
                .limit(1)
            ).first()
            return hit is not None
    except Exception as exc:
        log.warning(
            "already_replied lookup failed for %r (%s: %s); assuming NOT "
            "replied and continuing", reply_to, type(exc).__name__, exc,
        )
        return False
