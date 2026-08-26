"""
Ingest CLI — Phase 2 entrypoint. Fetches recent messages from Gmail, runs
each through the checkpointed LangGraph pipeline. Each message pauses at
the interrupt after persist_pending; the FastAPI service is where humans
approve/reject to advance them.

Usage:
    python -m app.cli.ingest

Two idempotency guards run before any LLM call (unchanged from Phase 1):
  1. gmail_id fast-path — skips before hitting Gmail's get API.
  2. message_id canonical — catches same Message-ID under a different
     gmail_id (cross-account forwarding).

Under Phase 2, "already processed" now includes "awaiting_approval,
sent_to_gmail_drafts, rejected" — any of those means the thread has been
seen. Re-ingest does not disturb an in-flight approval.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal

from app import dead_letter
from app.candidate import load_profile
from app.config import settings
from app.db.engine import session_scope
from app.db.models import Message, MessageStatus, Run, RunStatus
from app.gmail.client import GmailClient
from app.gmail.parser import parse_message
from app.llm.client import LLMClient
from app.observability.tracing import trace_callbacks
from app.pipeline.checkpointer import open_checkpointer
from app.pipeline.graph import build_graph
from app.retry import PermanentExternalError

log = logging.getLogger(__name__)


def main() -> None:
    _configure_logging()

    profile = load_profile()
    gmail = GmailClient.create()
    llm = LLMClient()

    with session_scope() as s:
        run = Run(status=RunStatus.RUNNING)
        s.add(run)
        s.flush()
        run_id = run.id
    log.info("Opened run %s", run_id)

    counters = _empty_counters()
    ids = gmail.list_message_ids(settings.gmail_label, settings.gmail_max_messages)
    counters["fetched"] = len(ids)
    log.info("Fetched %d message ids from label=%s", len(ids), settings.gmail_label)

    # The checkpointer opens a psycopg connection; keep it open for the
    # whole run so every graph.invoke() shares one connection pool slot.
    with open_checkpointer() as checkpointer:
        graph = build_graph(llm, profile, gmail, checkpointer)

        try:
            for i, gmail_id in enumerate(ids, start=1):
                # CONCEPT: per-message try/except with dead-letter fallback.
                #   A single message's infra failure (LLM 401, Gmail 5xx after
                #   retry exhaustion) should not kill a run that could
                #   process the other 199 messages successfully. We catch
                #   PermanentExternalError specifically because retry.py has
                #   already decided this failure is unrecoverable; anything
                #   else is either a genuine bug in our code (should crash)
                #   or a domain failure the graph already handled by
                #   persisting extraction_failed.
                try:
                    _process_one(gmail_id, gmail, graph, run_id, counters)
                except PermanentExternalError as pex:
                    counters["dead_lettered"] += 1
                    # GOTCHA: at this point we may or may not have a
                    # parsed Message-ID — the failure could have happened
                    # inside gmail.get_message before parsing. Passing
                    # None is fine; the dead_letters table column is
                    # nullable for exactly this case.
                    dead_letter.record(
                        node=_node_from_pex(pex),
                        error=pex,
                        message_id=None,  # unknown without parse
                        run_id=run_id,
                    )
                    log.warning(
                        "gmail_id=%s dead-lettered after %d attempts: %s",
                        gmail_id, pex.attempts, pex.original,
                    )
                except Exception as exc:
                    # CONCEPT: quarantine the message, not the run.
                    #   The clause above was the whole guard, and it only
                    #   covered retry exhaustion. That left a gap for
                    #   failures caused by the CONTENT of a message rather
                    #   than by infrastructure — and content is the one input
                    #   this system treats as hostile by default.
                    # GOTCHA: on 2026-08-21 a recruiter mail carrying a NUL
                    #   byte produced psycopg DataError inside persist_terminal.
                    #   Not a PermanentExternalError, so it fell through to the
                    #   outer handler, which marks the run failed and re-raises.
                    #   One malformed email ended a 100-message run at #42,
                    #   discarding 58 unprocessed messages. D44 fixes the NUL
                    #   at the parser; this clause fixes the blast radius, so
                    #   the NEXT unanticipated content bug costs one message
                    #   instead of a run.
                    # ALTERNATIVE: keep letting non-retry errors crash, on the
                    #   argument that they are our bugs and should be loud.
                    #   Rejected once this runs unattended on a timer: a crash
                    #   loop on a permanently-malformed message is silence, not
                    #   loudness. log.exception preserves the full traceback and
                    #   the dead_letters row is queryable, so nothing is hidden
                    #   — it just no longer takes the batch down with it.
                    counters["dead_lettered"] += 1
                    log.exception(
                        "gmail_id=%s failed with unhandled %s; dead-lettering "
                        "and continuing with the rest of the run",
                        gmail_id, type(exc).__name__,
                    )
                    dead_letter.record(
                        node="ingest",
                        error=PermanentExternalError(
                            f"ingest: unhandled {type(exc).__name__}",
                            original=exc,
                            # WHY attempts=1: nothing retried this. Recording 1
                            # keeps the column honest rather than implying a
                            # retry policy ran and gave up.
                            attempts=1,
                            elapsed_ms=0,
                        ),
                        message_id=None,
                        run_id=run_id,
                    )
                if i % 10 == 0:
                    log.info(
                        "Progress %d/%d — awaiting=%d done=%d dead=%d",
                        i, len(ids), counters["awaiting_approval"],
                        counters["terminal"], counters["dead_lettered"],
                    )
            run_status = RunStatus.SUCCEEDED
        except Exception:
            log.exception("Run failed; recording partial counters")
            run_status = RunStatus.FAILED
            raise
        finally:
            _finalise_run(run_id, run_status, counters)

    print(
        f"\nRun {run_id} done."
        f" new={counters['new']}"
        f" awaiting_approval={counters['awaiting_approval']}"
        f" terminal={counters['terminal']}"
        f" auto_sent={counters['auto_sent']}"
        f" dead_lettered={counters['dead_lettered']}"
        f" cost=${counters['cost']:.4f}"
        f"\nApprove/reject via API: http://{settings.api_host}:{settings.api_port}/pending"
        f"\nOr view report:         python -m app.cli.report --latest"
        f"\nDaily digest:           python -m app.cli.digest"
    )


def _process_one(gmail_id, gmail, graph, run_id, counters) -> None:
    with session_scope() as s:
        if s.query(Message.gmail_id).filter_by(gmail_id=gmail_id).first():
            return

    raw = gmail.get_message(gmail_id)
    parsed = parse_message(raw)

    with session_scope() as s:
        if s.get(Message, parsed.message_id):
            return

    counters["new"] += 1

    # CONCEPT: thread_id = Message-ID. The `configurable` dict is
    # LangGraph's way of scoping a call to a thread. Every subsequent
    # invoke against the same thread_id (from ANY process) will resume
    # from the last checkpoint for this Message-ID.
    # TRACE: `callbacks` is empty unless LANGSMITH_TRACING is on. When it is,
    # this one tracer produces a run tree per message — a parent run for the
    # graph and a child per node — with every string passed through the
    # redactor in app/observability/redaction.py before upload.
    config = {
        "configurable": {"thread_id": parsed.message_id},
        "callbacks": trace_callbacks(),
    }
    final_state = graph.invoke({"parsed": parsed, "run_id": run_id}, config=config)

    _tally_usage(counters, final_state)

    status = final_state.get("final_status")
    if status == MessageStatus.AWAITING_APPROVAL:
        counters["awaiting_approval"] += 1
    elif status == MessageStatus.AUTO_SENT:
        counters["auto_sent"] += 1
        counters["terminal"] += 1
    else:
        counters["terminal"] += 1


def _tally_usage(counters, final_state) -> None:
    for key, node_name in (
        ("classify_usage", "classify"),
        ("extract_usage", "extract"),
        ("score_usage", "score"),
        # Phase 4: embed_jd is an LLM cost too. Same aggregation shape;
        # embed completion_tokens is always 0 (embeddings have no output
        # tokens) so the _out column stays at 0 by construction.
        ("embed_usage", "embed"),
    ):
        u = final_state.get(key)
        if u and (u.prompt_tokens or u.completion_tokens):
            counters[f"{node_name}_in"] += u.prompt_tokens
            counters[f"{node_name}_out"] += u.completion_tokens
            counters["cost"] += u.cost_usd()


def _empty_counters() -> dict:
    return {
        "fetched": 0, "new": 0,
        "awaiting_approval": 0, "terminal": 0,
        # Phase 5 — auto-sent (subset of terminal) + infra failures.
        "auto_sent": 0,
        "dead_lettered": 0,
        "classify_in": 0, "classify_out": 0,
        "extract_in": 0, "extract_out": 0,
        "score_in": 0, "score_out": 0,
        # Phase 4 — no _out counterpart because embeddings have no
        # completion tokens. The runs table mirrors this asymmetry.
        "embed_in": 0, "embed_out": 0,
        "cost": Decimal("0"),
    }


def _node_from_pex(pex: PermanentExternalError) -> str:
    """Extract the retry_external `node` label from the wrapper message.

    The retry decorator stamps its node string as the prefix of the
    PermanentExternalError message (`"{node}: ..."`). Recovering it
    here means we don't have to add a `node` attribute to the exception
    class — same information, one less field to keep in sync.
    """
    text = str(pex)
    if ":" in text:
        return text.split(":", 1)[0]
    return "unknown"


def _finalise_run(run_id, status: str, counters: dict) -> None:
    with session_scope() as s:
        run = s.get(Run, run_id)
        run.finished_at = datetime.now(timezone.utc)
        run.status = status
        run.messages_fetched = counters["fetched"]
        run.messages_new = counters["new"]
        run.classify_tokens_in = counters["classify_in"]
        run.classify_tokens_out = counters["classify_out"]
        run.extract_tokens_in = counters["extract_in"]
        run.extract_tokens_out = counters["extract_out"]
        run.score_tokens_in = counters["score_in"]
        run.score_tokens_out = counters["score_out"]
        run.embed_tokens_in = counters["embed_in"]
        run.estimated_cost_usd = counters["cost"]


def _configure_logging() -> None:
    settings.log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_dir / "ingest.log"),
        ],
    )


if __name__ == "__main__":
    main()
