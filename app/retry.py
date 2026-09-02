"""
Retry policy for external calls.

This module has one public surface — the `@retry_external` decorator —
and one custom exception, `PermanentExternalError`. Every LLM and Gmail
method the pipeline touches is wrapped with the decorator; whatever the
decorator gives up on becomes a PermanentExternalError, which the
ingest loop catches and dead-letters.

=============================================================================
CONCEPT: retryable vs. permanent.
=============================================================================
"Retry it" is only the right answer when the failure is likely to have
been caused by something transient (a rate limit that will reset, a
5xx that reflects the provider's state not our request, a connection
that dropped in-flight). Retrying a fundamentally malformed request
(auth failure, wrong schema, missing resource) just spends more tokens
to fail the same way.

We enumerate the RETRYABLE set explicitly (whitelist). Everything else
is permanent. This is deliberate: it's much easier to notice "hey, we
should probably retry this new error type" than to notice "hey, we're
silently retrying an error we should have surfaced." Whitelist-of-safe
matches the D14 outbound-validator philosophy of enumerating what's
allowed rather than trying to enumerate what isn't.

=============================================================================
CONCEPT: exponential backoff, and why the jitter matters more than the base.
=============================================================================
Naive exponential backoff (`sleep = base * 2**attempt`) has a classic
failure mode: every client that failed at the same moment retries at
the same moment. If a thousand clients all get 429'd at t=0, they all
sleep for `base` seconds, then all retry at t=base — reproducing the
overload that caused the original 429. Then all of them retry at
t=base+2*base, again in lockstep. The pileup goes on until the base
term dominates the herd.

Jitter breaks the synchronisation. We use "full jitter" (see AWS
Builder's Library "Timeouts, retries, and backoff with jitter"):

    sleep = uniform(0, min(cap, base * 2**attempt))

Every retry lands somewhere in [0, exp_bound] rather than exactly at
exp_bound. A thousand clients that failed together retry across a
window instead of a single point, so the recovering endpoint sees a
smoothed stream instead of another spike.

The cap prevents pathological waits at high attempt counts
(2**5 * 1s == 32s; 2**10 * 1s == 17 minutes). At 5 attempts with
base=1s cap=60s, the worst-case cumulative wait is bounded around 63s.

=============================================================================
CONCEPT: idempotency of the wrapped call matters.
=============================================================================
Retrying a POST that already succeeded but whose response was lost in
transit produces duplicate work. All the operations we retry here are
either idempotent (embed the same text → same vector) or the
duplicate cost is small (list message ids returns the same list; a
duplicate get_message is a wasted round-trip; a duplicate create_draft
produces a duplicate draft, still human-review-gated). Sending a reply
IS NOT idempotent — see the extra care taken in the send path, which
does NOT retry on ambiguous outcomes (e.g. read-timeout after the
request bytes were on the wire).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class PermanentExternalError(Exception):
    """A failure that the retry policy gave up on.

    Carries the original exception + the retry attempt count so the
    dead-letter writer can record what happened without losing context.
    """

    def __init__(
        self,
        message: str,
        *,
        original: BaseException,
        attempts: int,
        elapsed_ms: int,
    ) -> None:
        super().__init__(message)
        self.original = original
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

# WHY import inside the classifier rather than at module top:
# openai and googleapiclient are heavy imports; the retry module is used
# from a lot of call sites and shouldn't pay startup cost for libraries
# it may or may not need. Lazy import inside the classifier keeps module
# load fast. GOTCHA: this means the classifier's None-guards fire when
# an SDK isn't installed — we treat "no SDK" as "not this SDK's error
# type" (fall through to the permanent bucket for unknown errors).


def _openai_retryable_types() -> tuple[type[BaseException], ...]:
    try:
        import openai
    except ImportError:
        return ()
    # WHY these specific classes: the OpenAI SDK v1 raises typed
    # exceptions for each status class. RateLimitError = 429,
    # APITimeoutError = client-side timeout, APIConnectionError =
    # transport failure, InternalServerError = 5xx.
    candidates = [
        getattr(openai, name, None)
        for name in (
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
        )
    ]
    return tuple(c for c in candidates if c is not None)


def _googleapi_is_retryable(exc: BaseException) -> bool:
    # googleapiclient wraps HTTP status in HttpError. We treat 429 and
    # 5xx as retryable; everything else in HttpError (4xx auth/perm/
    # notfound/badrequest) is permanent.
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        return False
    if not isinstance(exc, HttpError):
        return False
    status = getattr(exc.resp, "status", 0) if getattr(exc, "resp", None) else 0
    return status == 429 or 500 <= status < 600


def _transport_retryable_types() -> tuple[type[BaseException], ...]:
    # WHY httpx and requests both: openai's SDK uses httpx under the
    # hood; googleapiclient uses httplib2 (which raises socket errors,
    # already caught by ConnectionError below). Being liberal here is
    # safer than under-catching — a legitimate connection reset should
    # retry.
    types: list[type[BaseException]] = [
        ConnectionError,
        TimeoutError,
    ]
    try:
        import httpx
        types.extend([httpx.ConnectError, httpx.ReadTimeout,
                      httpx.RemoteProtocolError])
    except ImportError:
        pass
    try:
        # D69: google-auth raises its OWN transport exception, and it does
        # not inherit from the builtin ConnectionError — the MRO is
        # TransportError → GoogleAuthError → Exception. So a DNS failure
        # while refreshing the Gmail token fell straight through to the
        # permanent bucket and dead-lettered on attempt 1, with no retry.
        # GOTCHA: this was not theoretical. On 2026-09-01 a DNS blip inside
        # the container ("Unable to find the server at oauth2.googleapis.com")
        # dead-lettered 57 consecutive messages in a single run — every one
        # of them recoverable, none of them retried.
        # WHY this is safe to add: RefreshError — the "your token's scopes
        # are wrong" failure — is a SIBLING of TransportError under
        # GoogleAuthError, not a subclass. Adding TransportError therefore
        # does not start retrying genuine auth failures, which would burn
        # five attempts to fail identically five times.
        from google.auth.exceptions import TransportError
        types.append(TransportError)
    except ImportError:
        pass
    return tuple(types)


def is_retryable(exc: BaseException) -> bool:
    """Return True if `exc` should be retried by the exponential backoff.

    WHY a function and not a set at module top: the SDKs' exception
    classes may not exist yet when this module is imported (test env
    without openai installed, e.g. — currently we always have it, but
    the graceful degradation costs nothing).
    """
    retryable = _openai_retryable_types() + _transport_retryable_types()
    if isinstance(exc, retryable):
        return True
    return _googleapi_is_retryable(exc)


def _is_domain_error(exc: BaseException) -> bool:
    """True when `exc` was raised from inside our own code (app.*).

    Domain errors — LLMValidationError, LLMRefusalError, quarantine
    failures — carry business semantics that a specific caller knows how
    to react to (extractor's retry-with-feedback loop, for one). They
    are NOT infra failures and must not be converted to
    PermanentExternalError by this decorator.

    WHY module-prefix check rather than an isinstance-of-registered-set:
    the alternative is an explicit list of domain classes maintained
    somewhere, which drifts. The convention "if it came from app.*, the
    caller owns it" is one line, self-maintaining, and matches the
    project layout.
    GOTCHA: if we ever move an infra-error wrapper class INTO app.*,
    this rule would silently pass it through. We haven't — all infra
    error types belong to third-party SDKs.
    """
    return type(exc).__module__.startswith("app.")


# ---------------------------------------------------------------------------
# Backoff computation — factored out so tests can assert bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """The knobs. Defaults are conservative for a workstation-tier workload."""

    max_attempts: int = 5
    base_s: float = 1.0
    cap_s: float = 60.0


def compute_backoff(attempt: int, policy: RetryPolicy,
                    rng: random.Random | None = None) -> float:
    """Return the seconds to sleep BEFORE `attempt` (0-indexed).

    attempt=0 → 0 seconds (first try, no wait).
    attempt=1 → uniform(0, min(cap, base * 2)).
    attempt=n → uniform(0, min(cap, base * 2**n)).

    Injectable rng lets tests pin the sequence deterministically.
    """
    if attempt <= 0:
        return 0.0
    r = rng or random
    exp_bound = policy.base_s * (2 ** attempt)
    upper = min(policy.cap_s, exp_bound)
    return r.uniform(0.0, upper)


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------


def retry_external(
    *,
    node: str,
    policy: RetryPolicy | None = None,
    _sleep: Callable[[float], None] = time.sleep,
    _rng: random.Random | None = None,
) -> Callable[[F], F]:
    """Wrap an external call with retry-with-full-jitter on retryable errors.

    Args:
        node: a short label ("classify", "gmail_send") recorded on the
              PermanentExternalError. Used by the dead-letter writer to
              tag which surface failed.
        policy: retry knobs. Defaults to RetryPolicy().
        _sleep, _rng: injectables for tests. Do not set in production.

    Semantics:
        - Retryable exception → sleep, retry, up to max_attempts total.
          On exhaustion, raise PermanentExternalError wrapping the last one.
        - Permanent exception → raise PermanentExternalError immediately,
          wrapping the exception. No retry.
        - Success → return the wrapped call's result.

    Notes on non-retryable-by-design:
        Gmail send_reply passes max_attempts=1 (or similar) to disable
        retry — sending is not idempotent under ambiguous timeouts, and
        the human would rather see a dead-letter than a duplicate reply
        to the recruiter.
    """
    p = policy or RetryPolicy()

    def deco(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.monotonic()
            last: BaseException | None = None
            for attempt in range(p.max_attempts):
                wait = compute_backoff(attempt, p, rng=_rng)
                if wait > 0:
                    log.info(
                        "retry %s attempt %d/%d after %.2fs",
                        node, attempt + 1, p.max_attempts, wait,
                    )
                    _sleep(wait)
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if _is_domain_error(exc):
                        # Domain errors (LLMValidationError, etc.) belong
                        # to the caller — pass through unchanged so its
                        # own logic (e.g. extract's retry-with-feedback)
                        # can react. The retry policy is for INFRA only.
                        raise
                    last = exc
                    if not is_retryable(exc):
                        # GOTCHA: on a permanent external error we do NOT
                        # retry — not even the first attempt. The whole
                        # point of classification is that retrying an
                        # auth failure gives you five auth failures in a
                        # row instead of one.
                        elapsed_ms = int((time.monotonic() - started) * 1000)
                        raise PermanentExternalError(
                            f"{node}: permanent error {type(exc).__name__}: {exc}",
                            original=exc,
                            attempts=attempt + 1,
                            elapsed_ms=elapsed_ms,
                        ) from exc
                    log.warning(
                        "retry %s attempt %d/%d failed: %s: %s",
                        node, attempt + 1, p.max_attempts,
                        type(exc).__name__, exc,
                    )
            # Fell out of the loop → exhausted attempts on a retryable
            # error. Wrap the last one; the dead-letter writer will
            # record class + message + attempt count.
            elapsed_ms = int((time.monotonic() - started) * 1000)
            assert last is not None
            raise PermanentExternalError(
                f"{node}: exhausted {p.max_attempts} attempts; last error "
                f"{type(last).__name__}: {last}",
                original=last,
                attempts=p.max_attempts,
                elapsed_ms=elapsed_ms,
            ) from last

        return wrapper  # type: ignore[return-value]

    return deco


__all__ = [
    "PermanentExternalError",
    "RetryPolicy",
    "compute_backoff",
    "is_retryable",
    "retry_external",
]
