"""
LangGraph nodes for Phase 4 dedup.

Two nodes live here — `embed_jd` and `dedup_check`. embed_jd computes an
embedding for the extracted opportunity's jd_text and writes it into
state. dedup_check reads that embedding and runs the pgvector nearest-
neighbour query against prior opportunities, writing candidates into
state for persist to turn into duplicate_flags rows.

Design rationale for splitting embed_jd and dedup_check into two nodes:

  - Different failure surfaces. embed_jd fails on the Azure embedding
    API (network / auth / quota); dedup_check fails on Postgres (schema
    / connection). Isolating them means an outage in one doesn't
    contaminate the other's diagnostics.

  - Different side effects. embed_jd is a paid external call and its
    result is worth caching in state; dedup_check is a DB read that we
    can re-run cheaply. Splitting keeps embedding's "spent tokens"
    accounting honest — you can tell exactly how much cost the dedup
    pass introduced by looking at classify/extract/score/embed usage in
    isolation.

  - Testability. `embed_jd` is deterministic given a fixed LLM mock;
    `dedup_check` is deterministic given a fixed DB state. Merging them
    would force every test to set up both mocks.

CONCEPT: this node inherits the D13 tool-free guarantee.
  `embed_jd` calls LLMClient.embed(), which has no tools= parameter and
  no function-calling surface (see D24 and the red-team test
  test_llm_client_embed_no_tool_surface). No matter what the recruiter
  email content says, running it through the embedding API cannot fire
  an HTTP request, invoke a tool, or mutate the database. The only
  observable effect is a 1536-dim vector arriving in state.
"""

from __future__ import annotations

import logging
import time

from app.candidate import CandidateProfile
from app.config import settings
from app.db.engine import session_scope
from app.dedup.embedder import embed_jd_text
from app.dedup.lookup import find_similar_opportunities
from app.llm.client import LLMClient, LLMError, ZERO_USAGE
from app.retry import PermanentExternalError
from app.pipeline.state import TriageState, make_event

log = logging.getLogger(__name__)


def make_embed_jd_node(llm: LLMClient, profile: CandidateProfile):
    """Factory for the embed_jd node.

    Skips embedding (returns empty updates) when any of:
      - DEDUP_ENABLED=false in settings
      - no opportunity in state (extraction failed or was skipped —
        should not happen since routing gates on this, but defence in
        depth)
      - opportunity.jd_text is None or below min_jd_chars
      - the embed call raises (logged, but doesn't fail the graph — a
        missed embedding produces a missed dedup flag, which per D28 is
        the cheaper failure)
    """
    min_chars = profile.dedup.min_jd_chars
    dedup_enabled = settings.dedup_enabled

    def n(state: TriageState) -> dict:
        started = time.monotonic()

        if not dedup_enabled:
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "embed_usage": ZERO_USAGE,
                "events": [
                    make_event(
                        "embed_jd",
                        outcome="skipped: DEDUP_ENABLED=false",
                        duration_ms=duration_ms,
                    )
                ],
            }

        opp = state.get("opportunity")
        if opp is None:
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "embed_usage": ZERO_USAGE,
                "events": [
                    make_event(
                        "embed_jd",
                        outcome="skipped: no opportunity",
                        duration_ms=duration_ms,
                    )
                ],
            }

        jd_text = opp.jd_text or ""
        if len(jd_text) < min_chars:
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "embed_usage": ZERO_USAGE,
                "events": [
                    make_event(
                        "embed_jd",
                        outcome=f"skipped: jd_text {len(jd_text)} chars < {min_chars} floor",
                        duration_ms=duration_ms,
                    )
                ],
            }

        try:
            vector, usage = embed_jd_text(jd_text, llm)
        except (LLMError, PermanentExternalError) as e:
            # WHY log-and-continue instead of raise: a missed embedding
            # only degrades dedup for this one message. The rest of the
            # pipeline (rules, score, draft) is entirely independent of
            # the embedding — no reason to fail the whole graph. Per D28
            # the failure mode here is "we won't flag potential
            # duplicates for this message," which is the cheaper tail.
            #
            # GOTCHA: PermanentExternalError is in this tuple because it is
            # NOT a subclass of LLMError — it derives straight from
            # Exception. `except LLMError` alone therefore misses the single
            # most likely real-world failure here: retry_external exhausting
            # its attempts against a 429 and raising PermanentExternalError.
            # That escaped this node, escaped graph.invoke(), and got the
            # whole message dead-lettered in the ingest loop — discarding a
            # classification and extraction already paid for, and breaking
            # the promise this docstring and graph.py both make. Observed on
            # a real ingest 2026-08-21; see D43.
            # ALTERNATIVE: catch bare `Exception`, as dedup_check does at the
            # bottom of this module. Rejected here because embed_jd calls our
            # own code (embed_jd_text) where an AttributeError or a shape bug
            # is a defect we want loud, not silently degraded into "dedup
            # skipped". dedup_check's catch-all is defensible because its body
            # is a DB round-trip whose failure modes are all environmental.
            duration_ms = int((time.monotonic() - started) * 1000)
            log.warning(
                "embed_jd failed for message_id=%s: %s",
                state["parsed"].message_id, e,
            )
            return {
                "embed_usage": ZERO_USAGE,
                "events": [
                    make_event(
                        "embed_jd",
                        outcome=f"error: {e}",
                        duration_ms=duration_ms,
                    )
                ],
            }

        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "jd_embedding": vector,
            "embed_usage": usage,
            "events": [
                make_event(
                    "embed_jd",
                    outcome=f"embedded {len(jd_text)} chars → {len(vector)}-dim",
                    duration_ms=duration_ms,
                )
            ],
        }

    return n


def make_dedup_check_node(profile: CandidateProfile):
    """Factory for the dedup_check node.

    Runs after embed_jd. Queries opportunities-joined-messages for the
    top-K nearest neighbours within `lookback_days`, filters to those
    above `similarity_threshold`, writes them to state as
    `duplicate_candidates`. persist_pending / persist_terminal turn each
    into a duplicate_flags row after the new opportunity gets an id.

    Skips (returns empty candidates) when any of:
      - DEDUP_ENABLED=false
      - state.jd_embedding is None (embed_jd was skipped or failed)

    DB errors: log-and-continue with empty candidates. Per D28 the
    failure mode for missing a flag is asymmetric-cheap compared to
    failing the whole thread — a missed duplicate becomes a re-read on
    the human's next batch, not a lost message.

    CONCEPT: this node runs BEFORE the new opportunity is inserted into
    the DB (persist happens after rules/score/draft). So the query is
    "what did we see BEFORE this message" — the newly-processed
    opportunity cannot self-match because it doesn't exist yet. That
    ordering is deliberate (D29): dedup surfaces PRIOR context to inform
    the human's approval decision on the CURRENT message, not the other
    way around. If dedup ran post-persist, a new opp would trivially
    match itself and the query would need a WHERE-NOT-self clause; the
    pre-persist shape avoids that gymnastics.
    """
    threshold = profile.dedup.similarity_threshold
    lookback_days = profile.dedup.lookback_days
    top_k = profile.dedup.max_candidates_returned
    dedup_enabled = settings.dedup_enabled

    def n(state: TriageState) -> dict:
        started = time.monotonic()

        if not dedup_enabled:
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "duplicate_candidates": [],
                "events": [
                    make_event(
                        "dedup_check",
                        outcome="skipped: DEDUP_ENABLED=false",
                        duration_ms=duration_ms,
                    )
                ],
            }

        vector = state.get("jd_embedding")
        if vector is None:
            # TRACE: embed_jd already logged the reason (dedup disabled,
            # no opportunity, jd too short, or API error). We propagate
            # the skip forward — no candidates, no flags.
            duration_ms = int((time.monotonic() - started) * 1000)
            return {
                "duplicate_candidates": [],
                "events": [
                    make_event(
                        "dedup_check",
                        outcome="skipped: no jd_embedding",
                        duration_ms=duration_ms,
                    )
                ],
            }

        try:
            with session_scope() as s:
                candidates = find_similar_opportunities(
                    s,
                    vector,
                    threshold=threshold,
                    lookback_days=lookback_days,
                    top_k=top_k,
                )
        except Exception as e:
            # WHY catch-all rather than a specific DBAPIError: any exception
            # from the lookup (connection dropped, extension missing on a
            # misconfigured DB, malformed vector) should degrade to "no
            # flags for this message" rather than fail the whole graph.
            # Log-and-continue posture matches embed_jd's error branch and
            # persist_pending will simply write no duplicate_flags rows.
            duration_ms = int((time.monotonic() - started) * 1000)
            log.warning(
                "dedup_check failed for message_id=%s: %s",
                state["parsed"].message_id, e,
            )
            return {
                "duplicate_candidates": [],
                "events": [
                    make_event(
                        "dedup_check",
                        outcome=f"error: {e}",
                        duration_ms=duration_ms,
                    )
                ],
            }

        duration_ms = int((time.monotonic() - started) * 1000)
        if candidates:
            outcome = (
                f"{len(candidates)} match(es) ≥ {threshold} "
                f"(top sim={float(candidates[0].similarity):.4f})"
            )
        else:
            outcome = f"no matches ≥ {threshold} in last {lookback_days}d"
        return {
            "duplicate_candidates": candidates,
            "events": [
                make_event(
                    "dedup_check",
                    outcome=outcome,
                    duration_ms=duration_ms,
                )
            ],
        }

    return n
