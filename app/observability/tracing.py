"""
Turning tracing on, and making sure both trace sources go through redaction.

What this module does and why it exists here: LangSmith tracing in this
application has TWO independent sources, and wiring only one of them is the
most likely way to build this feature and believe it works. This module owns
both, hands them the same redacting client, and returns None when tracing is
off so that every call site is a two-line change rather than a conditional.

CONCEPT: the two sources.
  1. GRAPH — LangGraph emits a run per node (ingest, classify, extract, …)
     with the state going in and coming out. This arrives via langchain-core's
     `LangChainTracer`, which we pass as a callback on graph.invoke().
  2. LLM — the Azure calls themselves: prompt, completion, token counts.
     These do NOT come from LangGraph. app/llm/client.py uses the raw
     `AzureOpenAI` SDK (not LangChain), so nothing traces them unless the
     client object is wrapped by `langsmith.wrappers.wrap_openai`.

GOTCHA — the failure this module exists to prevent:
  `wrap_openai` uses the GLOBAL LangSmith client unless it is handed one.
  Wire the graph with a redacting client and leave the wrapper on the default,
  and the result is a trace tree where node state is properly redacted while
  the raw prompt — the entire recruiter email, verbatim — sits in the child
  LLM run beside it. It looks correct in the UI at a glance. Both sources take
  `get_client()` below; there is no second client anywhere in this package.

CONCEPT: why tracing is off unless explicitly armed.
  An unset LANGSMITH_TRACING means get_client() returns None, no callback is
  attached, wrap_openai is never applied, and not one byte leaves the process.
  Tests therefore trace nothing without any special casing, and the default
  posture for a system handling other people's mail is "does not phone home".
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> Any | None:
    """The one LangSmith client, or None when tracing is disabled.

    TRACE: called by LLMClient.__init__ (to decide whether to wrap the Azure
    client) and by get_tracer() (to build the graph callback). Cached, so both
    receive the SAME object and therefore the same anonymizer.

    GOTCHA: lru_cache means flipping LANGSMITH_TRACING requires a process
    restart. That is correct for this setting and wrong for the kill switch —
    the difference is that a kill switch must take effect mid-run (D36) while
    "is this deployment observed" is a boot-time property. Do not copy this
    caching pattern to anything that gates an outbound action.
    """
    if not settings.langsmith_tracing:
        return None
    if not settings.langsmith_api_key:
        # WHY warn instead of raise: tracing is optional infrastructure. A
        # missing key should degrade to "not traced", never to "ingest does
        # not run". The same argument as the LLM drafter falling back to a
        # template rather than to no reply at all.
        log.warning(
            "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is unset; "
            "continuing WITHOUT tracing."
        )
        return None

    from langsmith import Client

    from app.observability.redaction import build_anonymizer

    client = Client(
        api_key=settings.langsmith_api_key,
        # CONCEPT: `anonymizer` runs on run INPUTS, OUTPUTS and ERROR before
        # upload. It is a transform, not a filter — the run still appears in
        # LangSmith with its timings, token counts and tree structure intact;
        # only the strings are rewritten. That is what makes strict redaction
        # tolerable: the shape of the run, which is what you debug routing
        # with, survives it.
        # GOTCHA — the scope of that list is exact and was verified against
        # langsmith 0.9.8, not assumed. `Client._anonymizer` is applied in
        # three places: json_inputs, json_outputs, and {"error": error}. It is
        # NOT applied to run metadata, tags, or the run name. Anything this
        # module stamps into those three channels is therefore uploaded
        # VERBATIM, which is why trace_metadata() below enforces the identifier
        # policy itself instead of delegating it. Do not add a field there on
        # the assumption the redactor will catch it — it will not.
        anonymizer=build_anonymizer(identifiers=settings.langsmith_trace_identifiers),
    )
    log.info(
        "LangSmith tracing ENABLED for project %r (identifiers=%s). Free-text "
        "fields are dropped, not masked, in both modes.",
        settings.langsmith_project,
        "VISIBLE — subject and sender upload unmasked"
        if settings.langsmith_trace_identifiers
        else "redacted",
    )
    return client


def get_tracer() -> Any | None:
    """A LangChainTracer bound to the redacting client, or None.

    Returned rather than installed globally so the call site stays explicit:
    app/cli/ingest.py adds it to the per-message invoke config next to
    thread_id, where anyone reading the invoke can see that tracing happens.
    """
    client = get_client()
    if client is None:
        return None

    from langchain_core.tracers.langchain import LangChainTracer

    return LangChainTracer(project_name=settings.langsmith_project, client=client)


def trace_callbacks() -> list[Any]:
    """Callbacks to merge into a graph invoke config. Empty when disabled.

    WHY a list rather than an Optional tracer: the call site can splat this
    unconditionally (`"callbacks": trace_callbacks()`), and an empty list is
    exactly what LangGraph expects for "no callbacks". No `if` at the call
    site means no branch that can be got wrong later.
    """
    tracer = get_tracer()
    return [tracer] if tracer is not None else []


# -----------------------------------------------------------------------------
# Making a run identifiable in the LangSmith list view
# -----------------------------------------------------------------------------
#
# CONCEPT: a trace you cannot find is a trace you do not have.
#   Strict redaction leaves every run in the project titled with LangGraph's
#   default name and carrying `[REDACTED:free-text]` where the subject was.
#   Correct, and unusable for the actual monitoring question — "is THIS email
#   in the right category?" — because you cannot tell the rows apart to pick
#   the one you mean. These three functions attach the three things LangSmith
#   surfaces in a list: a NAME (what you read), METADATA (what you filter on)
#   and TAGS (what you slice a whole project by).
#
# GOTCHA: all three bypass the anonymizer entirely (see get_client). The
#   policy is enforced here, in the builders, by simply not putting a value in
#   the dict when identifiers are off. There is no second net below this one.

# WHY truncate: the subject is a display string in a list view, and a
# 300-character forwarded subject pushes everything else off the row. 80 is
# about what the LangSmith run list shows before eliding.
_SUBJECT_DISPLAY_CHARS = 80


def _identifiers_on() -> bool:
    """Both flags must be true. Tracing off means nothing is uploaded at all,
    so the identifier setting alone can never leak anything on its own."""
    return bool(settings.langsmith_tracing and settings.langsmith_trace_identifiers)


def trace_run_name(parsed: Any) -> str | None:
    """The title of this message's run tree, or None to keep LangGraph's.

    Returns None rather than a placeholder when identifiers are off: a run
    named "triage" repeated 200 times is worse than the default, which at
    least names the graph.

    TRACE: consumed by graph.invoke(config={"run_name": ...}) in
    app/cli/ingest.py. LangGraph applies it to the ROOT run only; child node
    runs keep their node names, which is what you want — the tree reads
    "Priya Sharma — Senior AI/ML Engineer" → classify → extract → …
    """
    if not _identifiers_on():
        return None
    subject = (parsed.subject or "(no subject)").strip()
    if len(subject) > _SUBJECT_DISPLAY_CHARS:
        subject = subject[: _SUBJECT_DISPLAY_CHARS - 1].rstrip() + "…"
    sender = parsed.from_name or parsed.from_email or "unknown sender"
    return f"{sender} — {subject}"


def trace_metadata(parsed: Any, run_id: Any = None) -> dict[str, Any]:
    """Filterable key/values for this run. Empty dict when tracing is off.

    CONCEPT: metadata is the half of this that does the real work. LangSmith
    lets you filter a project on a metadata key, so stamping the ingest run id
    here is what turns "show me everything from the 09:15 cycle" into one
    query, and `email_from` is what turns "every mail this recruiter ever
    sent" into another.

    GOTCHA: `message_id` looks like new exposure and is not. LangGraph already
    copies `configurable.thread_id` — which IS the RFC 5322 Message-ID, by
    D18 — into run metadata on every trace, unredacted, and has since tracing
    was first wired. Naming it explicitly here changes nothing about what
    leaves the process and makes the correlation key visible to a reader
    instead of hidden in framework-stamped fields.
    """
    if get_client() is None:
        return {}
    meta: dict[str, Any] = {
        # Correlation keys: these three are what join a LangSmith run back to
        # the `messages`, `runs` and `agent_activity` rows in Postgres.
        "message_id": parsed.message_id,
        "gmail_id": parsed.gmail_id,
        # Configuration in force for this run. WHY on every message rather
        # than assumed from the project: these change between cycles, and a
        # trace that does not record which mode produced it cannot answer
        # "was this the run where auto-send was armed?".
        "auto_send_mode": settings.auto_send_mode,
        "draft_mode": settings.draft_mode,
        "gmail_label": settings.gmail_label,
    }
    if run_id is not None:
        # str() because metadata is JSON-serialised and UUID is not
        # JSON-native — the same reason _finalise_run stringifies Decimal.
        meta["run_id"] = str(run_id)
    if _identifiers_on():
        # The identifier release. Key names match IDENTIFIER_KEYS in
        # redaction.py so the two channels state one policy, even though
        # nothing re-checks these on the way out.
        meta["email_subject"] = parsed.subject
        meta["email_from"] = parsed.from_email
        meta["email_from_name"] = parsed.from_name
    return meta


def trace_tags() -> list[str]:
    """Project-wide slices. Never message-specific, never personal.

    WHY tags and metadata both: a LangSmith tag is a coarse filter you can
    click, and it must stay low-cardinality to be useful. Anything that varies
    per message belongs in metadata; anything that varies per deployment
    belongs here.
    """
    if get_client() is None:
        return []
    return [
        f"label:{settings.gmail_label}",
        f"send:{settings.auto_send_mode}",
        f"draft:{settings.draft_mode}",
    ]


# -----------------------------------------------------------------------------
# One library warning, silenced on purpose
# -----------------------------------------------------------------------------

_warning_filtered = False

# CONCEPT: why suppressing a third-party warning is defensible HERE and is
#   usually not. A warning filter is a promise that you have understood the
#   warning and that it cannot become a real signal. That promise is only
#   keepable when the filter is narrow enough to match one known artefact and
#   nothing else — which is why this matches the message text AND the
#   `field_name='parsed'` detail, rather than silencing UserWarning, or
#   pydantic, or "serializer warnings" as a class. Any OTHER Pydantic
#   serialization warning, including one about our own models, still surfaces.
#
# WHAT it silences, traced end to end on 2026-09-04:
#   app/llm/client.py structured_completion
#     -> langsmith/wrappers/_openai.py:336 parse        (the wrap_openai shim)
#       -> langsmith/run_helpers.py:1729 _handle_container_end
#         -> langsmith/wrappers/_openai.py:263 _process_chat_completion
#           -> pydantic/main.py model_dump()            <- warning issued here
#
#   `beta.chat.completions.parse()` returns a ParsedChatCompletion whose
#   `message.parsed` field is GENERIC — it holds our ClassificationResult,
#   FitScore or Opportunity. LangSmith dumps that response to build the trace
#   payload, and at dump time the generic resolves to NoneType, so Pydantic
#   reports finding a model where it expected None. The data is dumped
#   correctly; only the declared type is wrong.
#
# GOTCHA — the reason it looked frightening and was not: our own TriageState
#   also has a field called `parsed` (the ParsedMessage). The names collide by
#   pure coincidence, and the warning has nothing to do with our state, our
#   checkpoints, or our database. Verified by experiment, not by reading:
#   LANGSMITH_TRACING=false produces zero of these, true produces one per
#   structured call, same code path otherwise.
#
# WHY silence it at all rather than live with it: it fires once per structured
#   LLM call — three per message — so a 200-message cycle buries the log in
#   them. That is not cosmetic. The D78 foreign-key violations went unnoticed
#   for four cycles precisely because real errors were lost in routine noise.
#
# Upstream: reported to the langsmith SDK; remove this filter when a release
# fixes _process_chat_completion to dump the parameterised model.
_PARSED_GENERIC_WARNING = (
    r"Pydantic serializer warnings:[\s\S]*field_name='parsed'"
)


def silence_parsed_completion_warning() -> None:
    """Filter the one Pydantic warning wrap_openai provokes. Idempotent.

    TRACE: called from wrap_llm_client below, and therefore installed if and
    only if wrap_openai was actually applied — the exact condition under which
    the warning can occur. An untraced process installs no filters at all.
    """
    global _warning_filtered
    if _warning_filtered:
        return
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=_PARSED_GENERIC_WARNING,
        category=UserWarning,
    )
    _warning_filtered = True
    log.debug(
        "filtered the ParsedChatCompletion serializer warning raised by "
        "langsmith's wrap_openai; see app/observability/tracing.py"
    )


def wrap_llm_client(azure_client: Any) -> Any:
    """Wrap the Azure SDK client for tracing, or hand it straight back.

    TRACE: called once, from LLMClient.__init__. When tracing is off this is
    the identity function and the object returned is the plain AzureOpenAI
    client, so the LLM path has zero added indirection in normal operation.

    GOTCHA: `tracing_extra={"client": ...}` is the whole point of this
    function. Without it wrap_openai builds its own client from the ambient
    environment — one with no anonymizer — and prompts upload unredacted. See
    the module GOTCHA.
    """
    client = get_client()
    if client is None:
        return azure_client

    from langsmith.wrappers import wrap_openai

    # Installed here, next to the thing that causes it, so the two can never
    # drift apart: no wrap_openai, no filter.
    silence_parsed_completion_warning()

    try:
        return wrap_openai(azure_client, tracing_extra={"client": client})
    except Exception:
        # WHY swallow: an SDK-version mismatch between openai and langsmith
        # must not stop the agent from answering recruiters. Losing LLM-level
        # traces is an inconvenience; losing the ingest cycle is an outage.
        log.exception(
            "wrap_openai failed; continuing with an untraced LLM client."
        )
        return azure_client
