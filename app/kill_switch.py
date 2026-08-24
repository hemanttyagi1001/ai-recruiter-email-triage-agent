"""
Kill switch — one function to check the halt flag, one CLI to flip it.

CONCEPT: why the check location matters more than the storage.
  Storing "sends are halted" in a DB row is trivial engineering. The
  hard part is WHERE the check fires.

    - Read at process start: a flip requires restart to take effect.
      Useful for permanent policy, useless for a kill switch.
    - Read when the Gmail client is constructed: same problem —
      captured at boot.
    - Read with a TTL cache (say, every 30 seconds): "flip and wait
      up to 30 seconds" is not a kill switch, it's a rate limiter.
    - Read IMMEDIATELY BEFORE the outbound HTTP call: the check
      reflects the flag's current value. Cost is one round-trip to a
      DB we already have connected. Fires only on the send path, not
      on read paths, so aggregate load is minimal.

  The last option is what a kill switch is FOR: an operator types
  `python -m app.cli.halt --on` and the very next send fails. No
  process restart, no cache flush, no coordinator. See D36.

CONCEPT: fail-safe vs. fail-open on the flag read itself.
  If the DB is unreachable when we try to read the flag, what should
  we do? Two options:
    - Fail-open: proceed with the send. "The system was working
      before, DB is briefly down, don't stop everything."
    - Fail-safe: refuse the send. "We can't verify the operator's
      current intent, so don't act."
  We fail-safe. The kill switch's job is to STOP outbound action; a
  transient DB outage during the read window is exactly when we
  shouldn't be sending mail whose approval status we can't verify.
  Cost of fail-safe: a temporary halt during a DB blip, all messages
  end up in awaiting_approval, human reviews after the outage ends.
  Cost of fail-open: an operator's halt can be silently ignored by a
  DB blip they don't know is happening. Fail-safe every time.
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.engine import session_scope
from app.db.models import SystemFlag

log = logging.getLogger(__name__)

HALT_KEY = "sends_halted"


def is_send_halted(session: Session | None = None) -> bool:
    """Return True if outbound Gmail action is halted.

    Args:
        session: optional session. If provided, the caller owns the
            transaction (used inside nodes that are already in a
            session_scope). If None, we open a fresh session.

    Semantics on failure to read (fail-safe): returns True (halted).
    See module docstring for the reasoning.
    """
    try:
        if session is not None:
            row = session.execute(
                select(SystemFlag).where(SystemFlag.key == HALT_KEY)
            ).scalar_one_or_none()
            return _row_to_halt(row)
        with session_scope() as s:
            row = s.execute(
                select(SystemFlag).where(SystemFlag.key == HALT_KEY)
            ).scalar_one_or_none()
            return _row_to_halt(row)
    except Exception as e:
        # GOTCHA: fail-safe. See module docstring. A DB blip becomes
        # "no sends until DB is back," which is the right behaviour.
        log.error(
            "is_send_halted read failed (%s: %s); assuming HALTED for safety",
            type(e).__name__, e,
        )
        return True


def _row_to_halt(row: SystemFlag | None) -> bool:
    # WHY missing row → halted: a bootstrapping error where the
    # migration didn't seed the row is exactly the kind of "we can't
    # verify operator intent" case we should fail-safe on. Migration
    # 0005 does insert the row, so this path is defence-in-depth.
    if row is None:
        log.warning(
            "system_flags row for %r missing; assuming HALTED for safety",
            HALT_KEY,
        )
        return True
    return bool(row.value)


def set_halt(*, value: bool, by: str | None = None) -> None:
    """Flip the halt flag. Used by app/cli/halt.py.

    updated_at is refreshed via SystemFlag.onupdate.
    """
    with session_scope() as s:
        row = s.execute(
            select(SystemFlag).where(SystemFlag.key == HALT_KEY)
        ).scalar_one_or_none()
        if row is None:
            # Should not happen — migration seeds the row. Insert as
            # recovery rather than raise; the operator's intent is to
            # halt (or unhalt), not to debug the migration.
            row = SystemFlag(key=HALT_KEY, value=value, updated_by=by)
            s.add(row)
        else:
            row.value = value
            row.updated_by = by


# ---------------------------------------------------------------------------
# CLI: python -m app.cli.halt --on|--off [--by "hemant"]
# The module lives at app.kill_switch but the CLI entrypoint lives at
# app.cli.halt so it's discoverable next to ingest / digest.
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Flip the outbound-send kill switch.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--on", action="store_true", help="Halt all outbound sends.")
    grp.add_argument("--off", action="store_true", help="Resume outbound sends.")
    grp.add_argument("--status", action="store_true",
                     help="Print current halt state and exit.")
    p.add_argument("--by", type=str, default=None,
                   help="Operator name for the audit trail.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.status:
        halted = is_send_halted()
        print(f"sends_halted = {halted}")
        return 0

    new_value = args.on  # --off means new_value=False (args.on is False)
    set_halt(value=new_value, by=args.by)
    print(f"sends_halted set to {new_value} (by {args.by or 'unknown'})")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
