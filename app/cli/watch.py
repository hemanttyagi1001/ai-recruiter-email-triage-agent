"""
Continuous poll loop — Phase 6 entrypoint, and the process the container runs.

What this module does and why it exists here: `app.cli.ingest` processes the
mailbox exactly once and exits, which is the right shape for a CLI a human
invokes. Running unattended needs something that keeps doing that on a timer,
survives a bad cycle, and stops cleanly when Docker asks it to. That is all
this module is — a supervisor around the existing ingest, deliberately holding
no pipeline logic of its own.

Usage:
    python -m app.cli.watch              # honours POLL_INTERVAL_MINUTES
    python -m app.cli.watch --once       # one cycle, then exit (smoke test)

CONCEPT: why a loop in the process rather than cron or a scheduler library.
  Cron would need a second process manager inside the container and would
  happily start a second ingest while the first is still running — Gmail
  fetches and LLM calls are slow enough for that to overlap. APScheduler or
  Celery would bring a dependency and a threading model for something a
  while-loop expresses exactly. A single-threaded loop makes overlap
  impossible by construction: the next cycle cannot begin until the previous
  one returns.

CONCEPT: why interrupting mid-cycle is safe.
  `docker stop` sends SIGTERM. We set a flag and finish the message in flight
  rather than dying instantly. Even a hard kill is survivable: every message
  is a checkpointed LangGraph thread keyed by Message-ID, so an interrupted
  run resumes from its last checkpoint, and the two idempotency guards in
  ingest skip anything already persisted. This is the payoff for the
  PostgresSaver work — the loop is restartable because the unit of work is.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from types import FrameType

from app import activity
from app.cli.ingest import main as ingest_once
from app.config import settings
from app.gmail.client import preflight_scopes

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Same shape as ingest's, but writing to watch.log.

    WHY its own file rather than reusing ingest.log: this process runs for
    weeks and interleaves supervisor events (cycle boundaries, shutdown) with
    per-run detail. Keeping the supervisor's own narrative separate makes
    "when did it restart and what mode was it in" answerable without reading
    every LLM call in between.
    """
    settings.log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_dir / "watch.log"),
        ],
    )

# GOTCHA: module-level rather than passed around because a signal handler runs
# outside normal call flow — it cannot return a value to the loop, so the flag
# it sets has to be reachable from both.
_shutdown_requested = False

# How finely we slice the wait between cycles. A single long sleep() would
# ignore SIGTERM for up to the full interval, and Docker only waits ~10s before
# escalating to SIGKILL — so a 15-minute sleep would guarantee every container
# stop became a hard kill.
_SLEEP_SLICE_SECONDS = 2


def _request_shutdown(signum: int, _frame: FrameType | None) -> None:
    global _shutdown_requested
    name = signal.Signals(signum).name
    if _shutdown_requested:
        # Second signal — the operator is insisting. Honour it.
        log.warning("%s received again; exiting immediately", name)
        sys.exit(130)
    _shutdown_requested = True
    log.info("%s received; finishing the current cycle then exiting", name)


def _sleep_interruptibly(seconds: float) -> None:
    """Sleep, but wake within _SLEEP_SLICE_SECONDS of a shutdown request."""
    remaining = seconds
    while remaining > 0 and not _shutdown_requested:
        nap = min(_SLEEP_SLICE_SECONDS, remaining)
        time.sleep(nap)
        remaining -= nap


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Gmail on an interval.")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle and exit. Useful for verifying the container "
             "is wired correctly without waiting for a second tick.",
    )
    parser.add_argument(
        "--interval-minutes", type=int, default=None,
        help="Override POLL_INTERVAL_MINUTES for this process.",
    )
    args = parser.parse_args()

    # GOTCHA: configure logging FIRST, before any log call in this function.
    # Handlers are installed by ingest's _configure_logging(), which does not
    # run until the first cycle — so without this, every line the supervisor
    # emits at startup goes nowhere. That silently swallowed the
    # AUTO_SEND_MODE=on warning below, which is the single line an operator
    # most needs to find when reconstructing why an email went out.
    # logging.basicConfig no-ops once handlers exist, so ingest's later call
    # is harmless; this one wins because it is first.
    _configure_logging()

    interval_minutes = args.interval_minutes or settings.poll_interval_minutes
    interval_seconds = interval_minutes * 60

    # SIGTERM is what `docker stop` sends; SIGINT is Ctrl-C. Both mean "stop
    # soon", and both deserve a clean exit rather than a traceback.
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    log.info(
        "watch starting: interval=%dm auto_send_mode=%s label=%s max_messages=%d",
        interval_minutes, settings.auto_send_mode,
        settings.gmail_label, settings.gmail_max_messages,
    )

    # D69: state the mailbox capability at boot rather than discovering it in
    # a traceback one interval later.
    # WHY we keep looping even when this returns False: the token file is a
    # bind mount. An operator fixing it on the host takes effect on the next
    # cycle with no restart, so exiting here would turn a self-healing
    # situation into one that needs a deploy. The error line is the signal.
    scopes_ok = preflight_scopes(settings.gmail_token_path)

    # WHY these two rows: they are the answer to "when did it last restart, and
    # in what mode" — the first question worth asking about a process that has
    # been running unattended for weeks, and one no other table records.
    activity.record(
        node="watch",
        event="watch_started",
        outcome=(
            f"interval={interval_minutes}m mode={settings.auto_send_mode} "
            f"label={settings.gmail_label}"
        ),
        detail={
            "interval_minutes": interval_minutes,
            "auto_send_mode": settings.auto_send_mode,
            "label": settings.gmail_label,
            "max_messages": settings.gmail_max_messages,
        },
    )
    activity.record(
        node="gmail",
        event="scopes_ok" if scopes_ok else "scopes_missing",
        level="info" if scopes_ok else "error",
        outcome=(
            "all required Gmail scopes present" if scopes_ok
            else "a REQUIRED Gmail scope is missing; every cycle will fail "
                 "until token.json is re-consented"
        ),
    )
    if settings.auto_send_mode == "on":
        # WHY this is a warning and not an info: it is the single most
        # consequential thing about this process. Someone reading logs after a
        # surprising email needs to find the moment sending was armed.
        log.warning(
            "AUTO_SEND_MODE=on — replies will be sent to recruiters with no "
            "human approval. The kill switch (app.cli.halt --on) still stops "
            "sends without restarting this process."
        )

    cycle = 0
    while not _shutdown_requested:
        cycle += 1
        started = time.monotonic()
        log.info("cycle %d starting", cycle)
        try:
            ingest_once()
        except Exception:
            # CONCEPT: a supervisor that dies on a bad cycle is not a
            # supervisor. ingest already dead-letters per-message failures;
            # anything reaching here is a whole-run failure — Gmail auth
            # expired, Postgres down, Azure returning 401. All of those are
            # usually transient or externally fixable, and none is improved by
            # the loop exiting. We log with traceback and try again next tick.
            # GOTCHA: this WILL spin quietly on a permanent failure such as a
            # revoked refresh token. The interval bounds it to one noisy
            # traceback per cycle rather than a hot loop, and the log line is
            # the signal to go look.
            log.exception("cycle %d failed; continuing to next cycle", cycle)
            # GOTCHA: this is the row that would have made D69's three-day
            # outage visible on day one. ingest's own run_finished row is
            # written in a finally block and so covers most failures, but a
            # crash BEFORE the run row exists — GmailClient.create() raising,
            # exactly what happened — produces nothing at all without this.
            exc = sys.exc_info()[1]
            activity.record(
                node="watch",
                event="cycle_failed",
                level="error",
                outcome=f"cycle {cycle}: {type(exc).__name__}: {exc}"[:500],
                detail={"cycle": cycle, "error_class": type(exc).__name__},
            )

        elapsed = time.monotonic() - started
        log.info("cycle %d finished in %.1fs", cycle, elapsed)

        if args.once:
            log.info("--once given; exiting after one cycle")
            break
        if _shutdown_requested:
            break

        # WHY subtract elapsed: the interval is meant to be cycle-to-cycle, not
        # gap-between-cycles. A 12-minute run followed by a full 15-minute
        # sleep would drift to a 27-minute period. If a cycle overruns the
        # interval entirely, the next one starts immediately.
        wait = max(0.0, interval_seconds - elapsed)
        if wait:
            log.info("sleeping %.0fs until cycle %d", wait, cycle + 1)
            _sleep_interruptibly(wait)

    log.info("watch stopped after %d cycle(s)", cycle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
