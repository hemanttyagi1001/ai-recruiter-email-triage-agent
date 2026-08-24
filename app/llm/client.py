"""
The single surface for every Azure OpenAI call in this codebase. Two things
are load-bearing about that "single":

1. Trust boundary. `structured_completion` and `embed` are the *only*
   methods exposed. Neither takes a `tools=` parameter, exposes function-
   calling, or offers an MCP surface. Email-facing modules (classify,
   extract, fit-score, embed_jd) can call this without inheriting the
   ability to grant the model action capability. See D13; the same
   argument extends to embeddings — an embedding call returns a vector,
   nothing else can happen from within it. See D24.

2. Cost accounting. Every LLM call funnels through here so the caller
   always gets a Usage record back — token counts, model name, cost helper.
   Aggregating into `runs.estimated_cost_usd` becomes bookkeeping instead
   of instrumentation.

Usage type carries the model name alongside the token counts because pricing
varies per deployment; the accountant at the run boundary doesn't have to
guess which model produced which tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeVar

from openai import AzureOpenAI
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.pricing import estimate_cost
from app.retry import retry_external

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# The only temperature a reasoning-model deployment will accept. Used to
# decide whether dropping the caller's value is worth warning about: a
# caller asking for 1.0 loses nothing, a caller asking for 0.0 does.
_REASONING_MODEL_FIXED_TEMPERATURE = 1.0

# WHY module-level and not per-instance: the condition being reported is a
# property of the deployment, not of any one client object. Once per process
# is the right cadence — a 200-message run makes ~600 completion calls and
# does not need 600 identical warnings.
_temperature_warning_emitted = False


# --- Errors ---


class LLMError(Exception):
    """Base class for LLM-layer failures."""


class LLMRefusalError(LLMError):
    """The model refused to answer (safety filter, policy)."""


class LLMValidationError(LLMError):
    """The model returned syntactically valid JSON that failed our Pydantic
    validators. Carries the usage so the caller still accounts for tokens
    spent on the failed call, and the raw content for retry-with-feedback."""

    def __init__(self, message: str, usage: "Usage", raw_content: str) -> None:
        super().__init__(message)
        self.usage = usage
        self.raw_content = raw_content


# --- Usage record ---


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    model: str

    def cost_usd(self) -> Decimal:
        return estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)

    def merged_with(self, other: "Usage") -> "Usage":
        # WHY merge: the extractor makes up to 3 calls; each returns a Usage.
        # We want a single Usage to hand upward for cost aggregation. Model
        # name is taken from `other` — if merging across deployments this
        # would lose info, but Phase 0 uses one deployment throughout.
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            model=other.model or self.model,
        )


ZERO_USAGE = Usage(0, 0, "")


def _warn_temperature_dropped(requested: float) -> None:
    """Say once, loudly, that a determinism request could not be honoured.

    WHY this is a warning and not a silent no-op: temperature=0.0 was chosen
    at the call sites to make classification reproducible — same email, same
    category. A reasoning deployment cannot offer that. Structured Outputs
    still guarantees the response *shape*, so nothing crashes and nothing
    looks wrong; only the run-to-run stability of borderline cases quietly
    disappears. That is exactly the class of change that should not be
    discoverable only by noticing odd numbers months later.
    """
    global _temperature_warning_emitted
    if _temperature_warning_emitted:
        return
    _temperature_warning_emitted = True
    log.warning(
        "temperature=%s was requested but AZURE_OPENAI_SUPPORTS_TEMPERATURE is "
        "false, so no temperature is being sent. Deployment %r samples at its "
        "own default: identical inputs may produce different (still schema-"
        "valid) outputs across runs. Any accuracy measurement taken now has "
        "run-to-run variance in it.",
        requested,
        settings.azure_openai_deployment,
    )


# --- Client ---


class LLMClient:
    """Wraps the Azure OpenAI client. Structural-completion only."""

    def __init__(self) -> None:
        # GOTCHA: max_retries=0 is load-bearing, not a micro-optimisation.
        # openai-python retries twice by default AND honours the server's
        # Retry-After header, which Azure sets to 60s on a 429. Left at the
        # default, every one of retry_external's 5 attempts silently becomes
        # up to 3 HTTP calls with 60s waits between them — so a rate-limited
        # embedding blocks for ~15 minutes instead of the ~30s the policy in
        # app/retry.py describes. Observed exactly that during a 100-message
        # ingest on 2026-08-21; see D42.
        # WHY zero rather than tuning the SDK's policy: two retry layers can
        # only ever multiply. One owner for the behaviour, and it is
        # retry_external, because that is what dead-letters on exhaustion and
        # what the digest reports on.
        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            max_retries=0,
        )
        self._deployment = settings.azure_openai_deployment

    @retry_external(node="llm_structured_completion")
    def structured_completion(
        self,
        schema: type[T],
        messages: list[dict],
        temperature: float = 0.0,
    ) -> tuple[T, Usage]:
        """Call the model, parse the response into `schema`, return (parsed, usage).

        Raises:
            LLMRefusalError: model declined to answer.
            LLMValidationError: model output failed Pydantic validation.
            LLMError: any other failure (empty content, missing usage, etc.)
        """
        # CONCEPT: beta.chat.completions.parse does three things for us:
        #   1. Converts our Pydantic model into the strict JSON schema Azure
        #      wants (all fields required, additionalProperties:false, etc.)
        #   2. Invokes chat.completions.create under the hood.
        #   3. Runs response.choices[0].message.content through
        #      schema.model_validate_json for us, exposing the result as
        #      .parsed (or None + .content when validation failed).
        # The alternative is to build the response_format dict by hand and
        # call model_validate_json ourselves — more code, no benefit.
        #
        # GOTCHA: `temperature` is not universally accepted. Reasoning-model
        # deployments (GPT-5 family) return HTTP 400 for any value other than
        # their default, so the kwarg has to be absent rather than present-
        # and-set-to-1.0. We omit it instead of coercing because a request log
        # showing an explicit temperature=1.0 reads as a deliberate choice,
        # when the truth is that we had no choice available.
        # TRACE: callers still pass temperature=0.0 (classify.py:64,
        # fit.py:86). On a temperature-capable deployment that value is sent
        # as-is; on a reasoning deployment it is dropped and warned about
        # once. Intent stays visible at the call site either way, so flipping
        # the config flag restores determinism without touching those files.
        kwargs: dict = {
            "model": self._deployment,
            "messages": messages,
            "response_format": schema,
        }
        if settings.azure_openai_supports_temperature:
            kwargs["temperature"] = temperature
        elif temperature != _REASONING_MODEL_FIXED_TEMPERATURE:
            _warn_temperature_dropped(temperature)

        try:
            resp = self._client.beta.chat.completions.parse(**kwargs)
        except ValidationError as e:
            # GOTCHA: older openai-python versions bubble ValidationError
            # from parse() instead of setting parsed=None. We can't recover
            # usage here — the exception was raised before the response
            # object was built. Report zero and log so it's visible.
            raise LLMValidationError(str(e), Usage(0, 0, self._deployment), "") from e

        if resp.usage is None:
            raise LLMError("response missing usage; cannot account for cost")

        usage = Usage(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            model=resp.model or self._deployment,
        )
        choice = resp.choices[0]
        msg = choice.message

        if getattr(msg, "refusal", None):
            raise LLMRefusalError(msg.refusal)

        if msg.parsed is None:
            # Strict JSON came back but our Pydantic validators failed
            # (e.g. ctc_max < ctc_min). Raw JSON is in msg.content.
            raise LLMValidationError(
                f"parsed=None; validators rejected content: {msg.content}",
                usage=usage,
                raw_content=msg.content or "",
            )

        return msg.parsed, usage  # type: ignore[return-value]

    @retry_external(node="llm_embed")
    def embed(self, text: str) -> tuple[list[float], Usage]:
        """Embed `text` into a fixed-length dense vector.

        Returns (vector, usage). Vector length is model-dependent
        (1536 for text-embedding-3-small, 3072 for -large). The caller
        stores it in a pgvector column sized to match — dimension
        mismatch is a schema error, not a runtime one.

        WHY a distinct method rather than shoehorning into
        structured_completion: the embeddings API is a different Azure
        endpoint with a different response shape (a `data[].embedding`
        list of floats, not a chat completion). One method per response
        shape keeps each surface honest about what it returns.

        Trust-boundary claim: like structured_completion, this method
        has no `tools=` parameter. It cannot invoke anything. The only
        observable side effect is returning a vector and a Usage record.
        Verified by tests/red_team/test_llm_client_embed_no_tool_surface.
        """
        if settings.azure_openai_embedding_deployment is None:
            # GOTCHA: this fires at call time, not at import time. That's
            # deliberate — a run with DEDUP_ENABLED=false must not need
            # the embedding deployment to boot. We only complain if
            # something actually tried to embed.
            raise LLMError(
                "azure_openai_embedding_deployment is not set; cannot embed. "
                "Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env, or set "
                "DEDUP_ENABLED=false to skip embedding entirely."
            )

        deployment = settings.azure_openai_embedding_deployment
        resp = self._client.embeddings.create(model=deployment, input=text)

        if resp.usage is None:
            raise LLMError("embeddings response missing usage; cannot account for cost")
        if not resp.data:
            raise LLMError("embeddings response contained no data[]")

        vector = list(resp.data[0].embedding)
        # WHY: embedding responses have prompt tokens only; completion tokens
        # is 0 by definition. We keep the Usage shape uniform so the
        # aggregation code at the run boundary doesn't branch on call type.
        usage = Usage(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=0,
            model=resp.model or deployment,
        )
        return vector, usage
