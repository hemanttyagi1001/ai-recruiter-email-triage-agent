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
        # CONCEPT: `anonymizer` runs on inputs, outputs and metadata before
        # upload. It is a transform, not a filter — the run still appears in
        # LangSmith with its timings, token counts and tree structure intact;
        # only the strings are rewritten. That is what makes strict redaction
        # tolerable: the shape of the run, which is what you debug routing
        # with, survives it.
        anonymizer=build_anonymizer(),
    )
    log.info(
        "LangSmith tracing ENABLED for project %r with strict redaction "
        "(free-text fields are dropped, not masked).",
        settings.langsmith_project,
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
