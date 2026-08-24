"""
Tests for app/retry.py — the @retry_external decorator + classifier.

Covers:
  1. Retryable exception, recovers on Nth attempt.
  2. Retryable exception exhausts attempts → PermanentExternalError.
  3. Permanent-external exception → PermanentExternalError on first try.
  4. Domain error (app.*) → passes through unchanged.
  5. compute_backoff bounded by min(cap, base * 2**attempt).
  6. is_retryable classifies transport / rate / timeout correctly.

WHY the sleep is mocked: unit tests must not actually wait 63 seconds
for the worst-case retry sequence. The decorator accepts _sleep and
_rng injectables specifically for this.
"""

from __future__ import annotations

import random

import pytest

from app.retry import (
    PermanentExternalError,
    RetryPolicy,
    compute_backoff,
    is_retryable,
    retry_external,
)


# ---------------------------------------------------------------------------
# Fake exception types
# ---------------------------------------------------------------------------


# A "retryable" fake — inherits from ConnectionError, which retry.py's
# _transport_retryable_types includes as always-retryable.
class _FakeTransientError(ConnectionError):
    pass


# A "permanent-external" fake — pretend it's from a third-party SDK by
# giving its class an external-looking __module__. We synthesise the
# class in-test so retry.py's _is_domain_error (which checks
# module.startswith("app.")) returns False.
class _FakePermanentError(Exception):
    pass
_FakePermanentError.__module__ = "some_sdk.errors"


# A "domain" fake — lives in an app.* module so retry.py treats it as
# ours (should pass through unchanged).
class _FakeDomainError(Exception):
    pass
# Test module lives in tests.*, but retry.py checks type(exc).__module__.
# Force it to look like app.* for this test's purpose.
_FakeDomainError.__module__ = "app.fake"


# ---------------------------------------------------------------------------
# Decorator behaviour
# ---------------------------------------------------------------------------


def test_retryable_recovers_on_second_attempt():
    calls = {"n": 0}

    @retry_external(node="unit", policy=RetryPolicy(max_attempts=3),
                    _sleep=lambda s: None)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _FakeTransientError("nope")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2


def test_retryable_exhausts_attempts_raises_permanent():
    @retry_external(node="unit", policy=RetryPolicy(max_attempts=3),
                    _sleep=lambda s: None)
    def always_fails():
        raise _FakeTransientError("always")

    with pytest.raises(PermanentExternalError) as ei:
        always_fails()
    assert ei.value.attempts == 3
    assert isinstance(ei.value.original, _FakeTransientError)


def test_permanent_external_error_raised_on_first_attempt():
    calls = {"n": 0}

    @retry_external(node="unit", policy=RetryPolicy(max_attempts=5),
                    _sleep=lambda s: None)
    def bad_auth():
        calls["n"] += 1
        raise _FakePermanentError("401")

    with pytest.raises(PermanentExternalError) as ei:
        bad_auth()
    assert ei.value.attempts == 1, "permanent errors must not be retried"
    assert calls["n"] == 1


def test_domain_error_passes_through_unchanged():
    """LLMValidationError-shaped exceptions belong to their caller; the
    decorator must not wrap them or retry them."""
    @retry_external(node="unit", policy=RetryPolicy(max_attempts=3),
                    _sleep=lambda s: None)
    def raises_domain():
        raise _FakeDomainError("this is business logic")

    with pytest.raises(_FakeDomainError):
        raises_domain()


def test_domain_error_is_not_retried():
    """The whole point of passing domain errors through: the extractor's
    retry-with-feedback loop lives OUTSIDE the decorator. Retrying a
    domain error at this layer would eat all its attempts before the
    caller even sees the first one."""
    calls = {"n": 0}

    @retry_external(node="unit", policy=RetryPolicy(max_attempts=5),
                    _sleep=lambda s: None)
    def raises_domain():
        calls["n"] += 1
        raise _FakeDomainError("bad payload")

    with pytest.raises(_FakeDomainError):
        raises_domain()
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Backoff math
# ---------------------------------------------------------------------------


def test_compute_backoff_zeroth_attempt_is_zero():
    """First try never sleeps."""
    assert compute_backoff(0, RetryPolicy()) == 0.0


def test_compute_backoff_bounded_by_exponential():
    """Full-jitter formula: sleep ∈ [0, min(cap, base * 2**attempt)]."""
    policy = RetryPolicy(max_attempts=5, base_s=1.0, cap_s=60.0)
    rng = random.Random(0)
    for attempt in range(1, 6):
        upper = min(policy.cap_s, policy.base_s * (2 ** attempt))
        # Sample enough times to be confident of the bound; each draw
        # must land in [0, upper].
        for _ in range(50):
            wait = compute_backoff(attempt, policy, rng=rng)
            assert 0.0 <= wait <= upper, (
                f"attempt={attempt} wait={wait} not in [0, {upper}]"
            )


def test_compute_backoff_hits_cap_at_high_attempt():
    """At high attempts, exponential dominates and the cap kicks in."""
    policy = RetryPolicy(base_s=1.0, cap_s=10.0)
    # 2**5 = 32 which is > cap=10; upper bound is the cap.
    class _MaxRng:
        def uniform(self, a, b):
            return b
    assert compute_backoff(5, policy, rng=_MaxRng()) == 10.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_is_retryable_transport_errors():
    """ConnectionError and TimeoutError are always retryable per
    _transport_retryable_types. These are stdlib types, not SDK types,
    so the test doesn't depend on any external package."""
    assert is_retryable(ConnectionError("reset"))
    assert is_retryable(TimeoutError("deadline"))


def test_is_retryable_returns_false_for_unknown():
    """A bare Exception isn't retryable — the whitelist rule means
    unknown things fail permanent, not retry."""
    assert not is_retryable(Exception("mystery"))
    assert not is_retryable(ValueError("bad arg"))
