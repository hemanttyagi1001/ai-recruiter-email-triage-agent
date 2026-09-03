"""
One-off cleanup for bounces that were ingested before D79 existed.

What this module does and why it exists here: the pipeline change in D79 only
sees mail it has not processed yet. Every non-delivery report already in the
`messages` table has a row, so the ingest CLI's gmail_id guard skips it before
the graph ever runs — it will sit in the inbox forever. The observed mailbox
had 30 such messages. This script is the backfill, and it is a CLI rather than
a startup task on purpose: a bulk irreversible mutation of someone's mailbox
should be a thing a person runs deliberately, once, having read what it plans
to do.

CONCEPT: dry run is the default, and there is no way to skip the summary.
  `--apply` is required to touch anything. Without it the script prints every
  message it would trash and exits. This inverts the usual CLI convention
  where the verb does the thing, and that is deliberate: the cost of running
  this accidentally is other people's mail in the bin, and the cost of the
  extra flag is four keystrokes.

GOTCHA: the detection here is the SAME function the pipeline uses
  (app/rules/undeliverable.is_undeliverable), applied to rows already in the
  database rather than to freshly parsed mail. If it were reimplemented, the
  backfill could trash a class of message the live path would keep, and the
  two would drift apart silently. It is imported, never copied.

Usage:
    python -m app.cli.trash_bounces            # dry run, prints what it would do
    python -m app.cli.trash_bounces --apply    # actually trashes
"""

from __future__ import annotations

import argparse
import logging
import sys
from types import SimpleNamespace

from sqlalchemy import select

from app import activity
from app.db.engine import session_scope
from app.db.models import Message, MessageStatus
from app.gmail.client import GmailClient
from app.kill_switch import is_send_halted
from app.rules.undeliverable import is_undeliverable

log = logging.getLogger(__name__)


def _candidates() -> list[SimpleNamespace]:
    """Rows that the D79 detector would call a bounce.

    WHY re-run the detector instead of selecting on
    status='skipped_undeliverable': that status only exists on rows ingested
    AFTER D79. Every historical bounce was filed as skipped_wrong_category by
    the classifier, which is exactly the population this script exists to
    find. Selecting on the status would return nothing and look like success.
    """
    with session_scope() as s:
        rows = s.execute(
            select(
                Message.message_id, Message.gmail_id,
                Message.from_email, Message.subject, Message.status,
            )
        ).all()

    return [
        SimpleNamespace(
            message_id=r.message_id, gmail_id=r.gmail_id,
            from_email=r.from_email, subject=r.subject, status=r.status,
        )
        for r in rows
        # is_undeliverable reads only .from_email and .subject, so a row
        # projection satisfies it without loading whole message bodies.
        if is_undeliverable(r)
    ]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="actually trash the messages. Without this, prints and exits.",
    )
    args = ap.parse_args(argv)

    found = _candidates()
    if not found:
        print("No bounces found in the messages table. Nothing to do.")
        return 0

    print(f"{len(found)} non-delivery report(s) found:\n")
    for m in found:
        print(f"  {m.gmail_id}  {m.from_email:<38}  {(m.subject or '')[:52]}")

    if not args.apply:
        print(
            f"\nDRY RUN — nothing was touched. Re-run with --apply to move "
            f"these {len(found)} message(s) to Gmail's Trash, where Gmail "
            f"purges them after 30 days."
        )
        return 0

    # Same gate the live node honours. An operator who has pulled the switch
    # because the agent is misbehaving should not find a bulk script exempt
    # from it. See D79.
    if is_send_halted():
        print(
            "\nABORTED: the kill switch is on. Clear it with "
            "`python -m app.kill_switch --off` if you intend to run this."
        )
        return 1

    gmail = GmailClient.create()
    trashed = failed = 0
    for m in found:
        if gmail.trash_message(m.gmail_id):
            trashed += 1
            # WHY update the status: the row said skipped_wrong_category, which
            # was true but is no longer the whole story. Recording the real
            # reason keeps "how much of my inbox was my own failed outbound"
            # answerable after this script has removed the evidence from Gmail.
            with session_scope() as s:
                row = s.get(Message, m.message_id)
                if row is not None:
                    row.status = MessageStatus.SKIPPED_UNDELIVERABLE
        else:
            failed += 1

    activity.record(
        node="trash_bounces", event="backfill_completed",
        level="info" if failed == 0 else "error",
        outcome=f"trashed={trashed} failed={failed} of {len(found)}",
        detail={"trashed": trashed, "failed": failed, "found": len(found)},
    )
    print(f"\nTrashed {trashed} of {len(found)}. Failed: {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
