"""
LangGraph state schema for the Phase 2 pipeline.

This module is the single place the graph's shared-state contract is
declared. It's small on purpose; the interesting work is in the field-by-
field commentary. Read the comments if nothing else in this file — the
choice between an implicit REPLACE reducer and an explicit APPEND reducer
is the difference between an audit trail and a truncated log.

=============================================================================
CONCEPT: LangGraph state, mechanically.
=============================================================================
LangGraph runs a graph as a sequence of "supersteps." At each superstep,
one or more nodes execute concurrently. Each node returns a partial dict
of updates; the framework MERGES those updates into the shared state.

The merge semantics is per-field, controlled by a "reducer." Default
reducer: OVERWRITE (replace the value with what the node returned). To
change it, annotate the field with a reducer function via
`Annotated[T, reducer]`:

    events: Annotated[list[NodeEvent], operator.add]
    #                                  ^^^^^^^^^^^^ merge function

Now `{"events": [event_a]}` from node1 followed by `{"events": [event_b]}`
from node2 merges to `state["events"] == [event_a, event_b]` — because
`operator.add` on two lists is concatenation.

Without the annotation: `state["events"] == [event_b]`. Only the LAST
write survives, and you don't find out until an audit shows a truncated
history. This is *the* subtle LangGraph bug — write it down, test for it.

=============================================================================
CONCEPT: which fields reduce, which overwrite, and why the answer matters.
=============================================================================
Rule of thumb: use APPEND when multiple nodes contribute; use OVERWRITE
when exactly one node produces the value (or when overwrite is the
INTENDED semantics, as with human edits).

REPLACE is safer than APPEND when you're not sure — the wrong REPLACE
gives you last-write-wins, which is at least easy to notice. The wrong
APPEND on a scalar field can silently corrupt data (`add(str, str)` gives
you concatenation).

See the per-field commentary below.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict
from uuid import UUID

from app.dedup.lookup import DuplicateCandidate
from app.drafts.validator import ValidationVerdict
from app.gmail.parser import ParsedMessage
from app.llm.client import Usage
from app.llm.schemas import FitScore, Opportunity
from app.rules.engine_types import RuleVerdict


# -----------------------------------------------------------------------------
# Node execution event — the smallest audit unit
# -----------------------------------------------------------------------------


@dataclass
class NodeEvent:
    """A single node's execution record. Every node returns exactly one of
    these into the `events` list. Together they form the audit trail that
    tests (and humans) use to reconstruct what happened."""

    node: str
    at: datetime
    duration_ms: int
    outcome: str | None = None  # short human-readable summary


def make_event(node: str, outcome: str | None = None, duration_ms: int = 0) -> NodeEvent:
    return NodeEvent(
        node=node,
        at=datetime.now(timezone.utc),
        duration_ms=duration_ms,
        outcome=outcome,
    )


# -----------------------------------------------------------------------------
# The state schema
# -----------------------------------------------------------------------------


class TriageState(TypedDict, total=False):
    """Shared state across all Phase 2 pipeline nodes.

    Field commentary below is load-bearing — do NOT change a reducer
    annotation without reading the WHY."""

    # -------------------------------------------------------------------------
    # Inputs — set once by the CLI/API before graph.invoke, never overwritten.
    # -------------------------------------------------------------------------

    # WHY REPLACE (default): populated once from outside the graph. A second
    # write is a bug (re-entering ingest for the same thread). REPLACE
    # semantics make that bug visible — the new value stomps the old and
    # tests would catch value mismatches. APPEND would fail loudly with a
    # TypeError trying to `add(ParsedMessage, ParsedMessage)`; also OK, but
    # noise for a case that can't happen in practice.
    parsed: ParsedMessage
    run_id: UUID

    # -------------------------------------------------------------------------
    # Classify — one node writes exactly once.
    # -------------------------------------------------------------------------

    # WHY REPLACE: `classify` runs once per thread; its outputs are
    # single-shot. REPLACE is the default and correct.
    category: str
    classify_usage: Usage
    # The model's one-sentence justification for `category` (D76). Audit only
    # — no router reads it, and it must stay that way; see the GOTCHA in
    # app/pipeline/classify.py about where this string came from.
    # WHY it lives in state rather than only in the NodeEvent: state is what
    # the checkpoint stores, so "why was this called a newsletter" stays
    # answerable from a resumed thread months later, and it is what makes the
    # reason visible on the classify node's run in LangSmith.
    classify_reason: str | None
    # Confidence as the model reported it. Carried for the same reason — a
    # wrong category at 0.99 and a wrong category at 0.41 are different bugs.
    classify_confidence: float | None

    # -------------------------------------------------------------------------
    # Extract — one node, may retry internally but returns once.
    # -------------------------------------------------------------------------

    # WHY REPLACE: extract's retry loop is INSIDE the node — it accumulates
    # usage internally and returns a single Usage record. The graph sees one
    # write. Don't confuse "the LLM was called 3 times" with "the state
    # field was set 3 times" — the second doesn't happen.
    opportunity: Opportunity | None
    extraction_retries: int
    extraction_error: str | None
    extract_usage: Usage

    # The address a reply should actually go to, resolved deterministically
    # after extraction. None means no human is reachable (job-board alert) and
    # the message terminates before drafting.
    # WHY this lives in state rather than being recomputed where it is needed:
    # act and auto_send both send mail, and a value computed twice is a value
    # that can disagree with itself after a refactor. Resolving once and
    # carrying it also puts the decision in the checkpoint, so "why did this
    # go to that address" is answerable from the stored state alone.
    reply_to: str | None

    # True when the message body asks the recipient to send a CV. Decided
    # deterministically at extract time; consumed by act and auto_send to
    # decide whether to attach the resume, and by the draft node to word the
    # closing line accordingly.
    resume_requested: bool
    # D67: canonical field labels a screening form asked for, in the order it
    # asked. Set only on the questionnaire path; empty everywhere else. The
    # ordering is load-bearing — it is what makes the reply paste-able into
    # the recruiter's ATS field by field.
    questionnaire_fields: list[str]

    # True when this recruiter has already been sent a reply (D60). Terminates
    # the message before any scoring or drafting spend.
    already_replied: bool

    # "template" or "llm" — which drafter produced draft_body. Load-bearing,
    # not bookkeeping: _route_after_validate refuses to auto-SEND an llm body
    # (D56), so this field is what stands between an injected instruction and
    # a real outbound email.
    draft_source: str

    # Tokens spent by the LLM drafter, when draft_mode=llm. Absent on the
    # template path, which costs nothing.
    draft_usage: Usage

    # -------------------------------------------------------------------------
    # Rules — one write, may set draft prep fields as a side effect.
    # -------------------------------------------------------------------------

    # WHY REPLACE: rules node either sets `rule_verdict` (with reason and
    # draft_type=decline) or leaves it None. Single write per thread.
    rule_verdict: RuleVerdict | None

    # -------------------------------------------------------------------------
    # Score — set when rules pass and score runs.
    # -------------------------------------------------------------------------

    # WHY REPLACE: score node runs at most once. `fit_score_result.uncertain`
    # is the routing signal — it lives inside the value, not as a separate
    # field, so the routing function reads state["fit_score_result"].uncertain.
    fit_score_result: FitScore | None
    score_usage: Usage
    # WHY separate `needs_review` bool: it's redundant with
    # fit_score_result.uncertain, but persist_terminal reads it as its own
    # signal so we're not coupling the persist decision to fit_score's
    # internal representation.
    needs_review: bool

    # -------------------------------------------------------------------------
    # Embedding — populated by embed_jd (Phase 4).
    # -------------------------------------------------------------------------

    # WHY REPLACE: embed_jd runs at most once per thread. The vector is a
    # 1536-dim list[float] (text-embedding-3-small). Written into state
    # and later flushed to opportunities.jd_embedding by persist.
    # None here means: embedding was skipped (dedup disabled, no
    # opportunity, jd_text below min_jd_chars, or the embedding call
    # failed and we degraded gracefully — see app/dedup/nodes.py).
    jd_embedding: list[float] | None
    embed_usage: Usage

    # -------------------------------------------------------------------------
    # Dedup lookup — populated by dedup_check (Phase 4 step 3).
    # -------------------------------------------------------------------------

    # WHY REPLACE: dedup_check runs at most once per thread. Value is a
    # list — but a list produced whole by a single node, not accumulated
    # across nodes, so REPLACE is correct. If APPEND were the reducer
    # here, a graph shape that revisited the node (e.g. a retry loop we
    # don't have today) would silently double every candidate. REPLACE
    # makes that impossible.
    # GOTCHA on absence vs. empty list: the KEY missing from state means
    # "dedup_check did not run" (e.g. embed_jd errored, or the graph
    # short-circuited before dedup — no such path exists today, but
    # defence in depth). The key present with an empty list means
    # "dedup_check ran, found no matches above threshold." persist reads
    # state.get("duplicate_candidates", []) so both cases produce zero
    # duplicate_flags rows — the distinction lives in the events audit.
    duplicate_candidates: list[DuplicateCandidate]

    # -------------------------------------------------------------------------
    # Draft prep — populated by rules/score before draft runs.
    # -------------------------------------------------------------------------

    # WHY REPLACE, and why this is set by TWO nodes (rules + score):
    # rules sets these to ("decline", <rule reason>) when a rule fires;
    # score sets them to ("interested", None) or ("decline", <soft reason>)
    # when scoring completes without abstain. Only one of rules/score
    # populates them per thread (they're mutually exclusive by graph shape),
    # so there's no true "which write wins" issue. REPLACE is correct here.
    draft_type: str | None
    draft_reason: str | None

    # -------------------------------------------------------------------------
    # Draft body + validation — one write each.
    # -------------------------------------------------------------------------

    # WHY REPLACE, DELIBERATELY, on draft_body: the draft node writes it
    # once, and the API's approve endpoint may OVERWRITE it via
    # `approved_body` (a separate field) — NOT by re-writing draft_body.
    # Keeping the two fields distinct means we always have the original
    # generated draft plus the human-edited version, side by side for
    # audit. GOTCHA: if someone consolidates these into one field and
    # gives it an APPEND reducer thinking "edits should stack," you'd
    # send a concatenation of every revision. That is a silent, terrible
    # bug that regulators would find funny. Don't.
    draft_body: str | None
    draft_validation: ValidationVerdict | None

    # -------------------------------------------------------------------------
    # Human approval — set by the API, read by act.
    # -------------------------------------------------------------------------

    # WHY REPLACE: the API's approve/reject endpoint sets these exactly once
    # per thread. A second call to the same endpoint should fail at the
    # API layer (409 Conflict) before it ever hits the graph. If somehow
    # both approve AND reject calls slip through, REPLACE means the last
    # one wins, which is at least visible in the audit trail via the events
    # list. APPEND would concatenate approval reasons into a nonsense
    # composite string.
    approval_status: str | None       # "approved" | "rejected"
    approval_reason: str | None
    approved_body: str | None          # human-edited body, if any

    # -------------------------------------------------------------------------
    # Act — Gmail I/O output.
    # -------------------------------------------------------------------------

    # WHY REPLACE: act runs exactly once (after interrupt) and either
    # populates gmail_draft_id (approved path) or leaves it None (rejected
    # path). Retrying is not a supported operation in Phase 2 — the
    # checkpoint after act is terminal.
    gmail_draft_id: str | None

    # -------------------------------------------------------------------------
    # Phase 5 — autonomy path (rule-based decline auto-send).
    # -------------------------------------------------------------------------

    # WHY REPLACE on both: auto_send runs at most once per thread. It
    # either populates gmail_sent_id (kill switch off, send succeeded)
    # or sets halted=True (kill switch on) which the router uses to
    # detour into the human-approval path. Mutually exclusive by design.
    gmail_sent_id: str | None
    halted: bool

    # -------------------------------------------------------------------------
    # Terminal — set by whichever persist node closes out the thread.
    # -------------------------------------------------------------------------

    final_status: str  # matches app.db.models.MessageStatus values

    # -------------------------------------------------------------------------
    # THE reducer field — the only one with APPEND semantics.
    # -------------------------------------------------------------------------

    # WHY APPEND: every node returns exactly one NodeEvent into this list.
    # If this reducer were REPLACE (the default), the state after the last
    # node would contain ONLY the last node's event — the whole audit trail
    # would be lost. The restart test in tests/test_restart.py explicitly
    # asserts that a resumed thread's events list contains entries for
    # every node that ran, both before and after the interrupt.
    #
    # HOW APPEND WORKS: LangGraph calls the reducer function with (current,
    # incoming). `operator.add` on two lists is `current + incoming`, i.e.
    # concatenation. Because Python lists are mutable, the underlying value
    # in state is a NEW list each time (not mutated in place) — LangGraph's
    # snapshot semantics rely on this immutability at the reducer boundary.
    #
    # GOTCHA: if a node accidentally returns `{"events": event}` (bare
    # NodeEvent, not wrapped in a list), `add(list, NodeEvent)` raises
    # TypeError. Loud failure, good. Watch for it in code review.
    events: Annotated[list[NodeEvent], operator.add]


# -----------------------------------------------------------------------------
# Convenience — the whole state, keys and their expected types, one glance.
# -----------------------------------------------------------------------------

# GOTCHA: TypedDict `__annotations__` includes the Annotated wrapper for
# fields with reducers. If you introspect state schema for docs, use
# `typing.get_type_hints(TriageState, include_extras=False)` to strip
# reducer metadata.

__all__ = ["TriageState", "NodeEvent", "make_event"]
