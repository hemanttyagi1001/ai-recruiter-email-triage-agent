"""
One-shot repair: fill `drafts.reply_to_email` for rows written before D60.

Migration 0006 added the column but could not populate it — the recipient was
never stored, only used. Every historical draft therefore has NULL, and a NULL
never matches the dedup lookup, so those recruiters would be replied to a
second time.

WHY reconstruct rather than leave them NULL: the dedup check is only as good
as its history. Starting from an empty history means the first cycle after
D60 re-answers everyone the agent has already written to — which is precisely
the behaviour the feature exists to prevent, delivered on day one.

The reconstruction is exact, not a guess: `resolve_reply_target` is pure and
deterministic, and both of its inputs (the message row and the opportunity
row) are still on disk. Running it now yields the same address the send path
computed then.

Usage:
    python -m app.cli.backfill_reply_to            # report only
    python -m app.cli.backfill_reply_to --apply    # write
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import select

from app.db.engine import session_scope
from app.db.models import Draft, Message, Opportunity as OpportunityRow
from app.gmail.parser import ParsedMessage
from app.llm.schemas import Opportunity
from app.rules.reply_target import resolve_reply_target

log = logging.getLogger(__name__)


def _as_parsed(m: Message) -> ParsedMessage:
    """Rebuild just enough of a ParsedMessage for the resolver.

    Only from_email, subject and message_id are read by resolve_reply_target;
    the rest is filled to satisfy the dataclass.
    """
    return ParsedMessage(
        message_id=m.message_id, gmail_id=m.gmail_id, thread_id=m.thread_id,
        from_email=m.from_email, from_name=m.from_name, subject=m.subject,
        received_at=m.received_at, body_text=m.body_text or "",
        raw_headers=m.raw_headers or {},
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Write the values. Without this, only reports.")
    args = ap.parse_args()

    filled = unresolved = 0
    with session_scope() as s:
        rows = s.execute(
            select(Draft, Message, OpportunityRow)
            .join(Message, Message.message_id == Draft.message_id)
            .outerjoin(OpportunityRow, OpportunityRow.message_id == Draft.message_id)
            .where(Draft.reply_to_email.is_(None))
        ).all()

        for draft, msg, opp_row in rows:
            opp = None
            if opp_row is not None:
                # Only the two fields the resolver reads. Rebuilding the whole
                # Opportunity would mean mirroring every column and would fail
                # on any validator the stored row predates.
                opp = Opportunity.model_construct(
                    recruiter_email=opp_row.recruiter_email,
                    company=opp_row.company,
                    role_title=opp_row.role_title,
                )
            target = resolve_reply_target(_as_parsed(msg), opp)
            if target is None:
                unresolved += 1
                log.info("  UNRESOLVED  %s  (%s)", msg.from_email, draft.status)
                continue
            filled += 1
            log.info("  %-38s -> %s  [%s]", msg.from_email, target, draft.status)
            if args.apply:
                draft.reply_to_email = target

        if not args.apply:
            # session_scope commits on exit; discard the in-memory edits.
            s.rollback()

    log.info("")
    log.info("%d resolved, %d unresolved, %d total",
             filled, unresolved, filled + unresolved)
    log.info("apply=%s", args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
