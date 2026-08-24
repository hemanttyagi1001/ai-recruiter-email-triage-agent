"""
Per-model token pricing table. Numbers are USD per 1M tokens.

GOTCHA: Azure OpenAI deployment names are user-chosen and don't map 1:1 to
model names. The `model` field on a completion response is the deployment
name for Azure, not "gpt-4o-mini". Two ways to get accurate cost:
  (a) name your deployments after the model (best practice)
  (b) add your deployment name to PRICING below

GOTCHA (measured 2026-08-21, not documented anywhere): Azure is not even
self-consistent about what `resp.model` holds. The chat endpoint returns the
*versioned* model name — "gpt-5-mini-2025-08-07" — while the embeddings
endpoint returns the bare name — "text-embedding-3-small". So (a) above is
necessary but not sufficient for chat deployments. `_strip_version` below
normalises the versioned form back to the family name so one PRICING entry
covers every snapshot.

If the deployment is unknown, we fall back to the DEFAULT (gpt-4o-mini
rates) and log so a mis-attribution is at least visible.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal

log = logging.getLogger(__name__)

# Matches the "-2025-08-07" tail Azure appends to chat model names.
_VERSION_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _strip_version(model: str) -> str:
    """Reduce "gpt-5-mini-2025-08-07" to "gpt-5-mini".

    WHY this matters more than it looks: Global Standard deployments are
    auto-upgraded to newer snapshots by Azure on a rolling schedule. Keying
    PRICING on the full versioned string would work perfectly until the day
    that upgrade lands, then silently fall back to default rates with no
    symptom except wrong numbers in the digest — a bug that reports itself
    as "the cost column looks a bit off" six weeks later.
    ALTERNATIVE: enumerate every snapshot as its own PRICING key. Rejected:
    it needs a code change every time Azure ships a version, which is the
    same failure with extra steps.
    """
    return _VERSION_SUFFIX.sub("", model)

# (input $/1M, output $/1M). Update as Azure/OpenAI ships new tiers.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    # GPT-5 family, Global Standard rates (Azure list price, 2026-08-21).
    # GOTCHA: these are reasoning models. The output rate is billed on
    # *reasoning* tokens too, not just the visible answer — a one-line
    # classification measured 79 in / 152 out on gpt-5-mini, where the
    # equivalent gpt-4o-mini call emitted ~10. Budget on output, not input.
    # Data Zone deployments cost ~10% more; add separate keys if we ever
    # deploy one, since the deployment name won't distinguish them.
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    # Embeddings — priced per input token; output side is n/a for embeddings
    # (the response is a fixed-size vector, not a token stream). We keep the
    # tuple shape uniform and set the output rate to 0 so cost math still
    # works whether the caller passes 0 completion_tokens (correct) or
    # accidentally passes non-zero.
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

DEFAULT_RATES: tuple[float, float] = PRICING["gpt-4o-mini"]

_warned_unknown: set[str] = set()


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
    """Return USD cost for a call. Uses DEFAULT_RATES for unknown deployments."""
    rates = PRICING.get(_strip_version(model))
    if rates is None:
        if model not in _warned_unknown:
            log.warning(
                "Unknown model/deployment %r for pricing; using default gpt-4o-mini rates. "
                "Add it to app/llm/pricing.py to get accurate numbers.",
                model,
            )
            _warned_unknown.add(model)
        rates = DEFAULT_RATES
    in_rate, out_rate = rates
    cost = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000
    # Quantize to 6 decimal places — matches the runs.estimated_cost_usd column.
    return Decimal(f"{cost:.6f}")
