"""
Dead-letter writer — one function.

CONCEPT: what a dead-letter row means, and what it doesn't.
  A row in `dead_letters` means: retry_external gave up on an external
  call. That is, we tried enough times to be confident it's not
  transient, or the first attempt classified the error as permanent
  (auth failure, malformed request). The row is a durable record so
  that:
    (a) the daily digest can surface "you have N failures nobody
        looked at yet";
    (b) an operator investigating a run has one place to find infra
        failures instead of grepping logs; and
    (c) whoever fixes the underlying issue (rotate an API key, ask
        the provider to raise a quota) can point at a specific row
        as "this failure was the reason."

  A dead-letter row is NOT:
    - a signal to auto-recover. Nothing in the pipeline reads this
      table. Re-running ingest may re-produce the same failure; that's
      the operator's judgement.
    - a domain failure. Extract's "model kept returning garbage" lands
      on `messages.extraction_error` with status EXTRACTION_FAILED.
      Dead-letter is strictly infra.
    - complete provenance of a bad run. Logs still exist and carry the
      full retry sequence with jitter timings. This table carries the
      minimum needed to react (class, message, node, when).
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from app.db.engine import session_scope
from app.db.models import DeadLetter
from app.retry import PermanentExternalError

log = logging.getLogger(__name__)


def record(
    *,
    node: str,
    error: PermanentExternalError,
    message_id: str | None = None,
    run_id: UUID | None = None,
) -> UUID:
    """Persist a dead-letter row for `error`. Returns the new row's id.

    Uses its own session_scope — the caller is often in the middle of
    a broken transaction (the graph invoke that raised), and reusing
    that session would compound the failure. A fresh transaction is
    cheap and guarantees the dead-letter write survives whatever the
    caller does next.

    GOTCHA: this function must not itself raise on infrastructure
    failure. If the DB is down (a plausible correlated failure), we
    log-and-swallow — the alternative is a crash loop where the
    dead-letter writer fails to record its own failure. The log line
    is the last-resort audit trail in that case.
    """
    row_id = None
    try:
        with session_scope() as s:
            row = DeadLetter(
                message_id=message_id,
                run_id=run_id,
                node=node,
                # WHY the qualified class name (module.ClassName): the
                # bare class name "AuthenticationError" is ambiguous
                # across SDKs (openai has it, google has it). The fully-
                # qualified name is unambiguous and greppable.
                error_class=(
                    f"{type(error.original).__module__}."
                    f"{type(error.original).__name__}"
                ),
                error_message=str(error.original)[:2000],  # keep bounded
                error_details={
                    "attempts": error.attempts,
                    "elapsed_ms": error.elapsed_ms,
                    # Human-readable wrapper message from the retry helper.
                    "retry_message": str(error)[:2000],
                    # Timestamp for observability tools that parse JSONB
                    # without joining occurred_at back on the row.
                    "recorded_at": datetime.utcnow().isoformat() + "Z",
                },
            )
            s.add(row)
            s.flush()  # populate row.id before we exit the scope
            row_id = row.id
        log.warning(
            "dead_letter recorded: node=%s message_id=%s class=%s attempts=%d",
            node, message_id, type(error.original).__name__, error.attempts,
        )
        return row_id
    except Exception as write_exc:
        # WHY log-and-return: see GOTCHA above. Correlated infra failure
        # (DB is what died) must not crash the caller trying to record
        # it. The log is the fallback record.
        log.exception(
            "FAILED to write dead_letter for node=%s message_id=%s; "
            "original error was %s: %s; write error was %s: %s",
            node, message_id,
            type(error.original).__name__, error.original,
            type(write_exc).__name__, write_exc,
        )
        return row_id  # None; caller can distinguish success from failure
