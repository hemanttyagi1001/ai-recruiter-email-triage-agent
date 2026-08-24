"""
Retry-with-error-feedback behaviour of the extractor.

Cases:
  1. Validation error on attempt 1, valid on attempt 2 → recovery.
  2. Validation errors on all attempts → returns (None, MAX_RETRIES, error, usage)
     without raising. Usage from all failed attempts is summed — we paid for them.
"""

from __future__ import annotations

from app.llm.client import LLMValidationError
from app.llm.schemas import Opportunity
from app.pipeline.extract import MAX_RETRIES, extract


def test_extractor_recovers_after_one_validation_failure(fake_llm, usage_factory):
    fake_llm.queue(
        LLMValidationError("ctc_max < ctc_min", usage_factory(80, 15), "{}")
    )
    fake_llm.queue(
        (
            Opportunity(company="Acme", role_title="Engineer", ctc_min_lpa=25, ctc_max_lpa=35),
            usage_factory(95, 25),
        )
    )

    opp, retries, err, agg_usage = extract("Subject", "Body", fake_llm)

    assert opp is not None
    assert opp.company == "Acme"
    assert retries == 1
    assert err is None
    # Aggregated: both calls counted, even the failed one.
    assert agg_usage.prompt_tokens == 80 + 95
    assert agg_usage.completion_tokens == 15 + 25
    # Retry loop must have appended the error-feedback turns.
    _, last_messages = fake_llm.calls[-1]
    assert any("failed validation" in m["content"] for m in last_messages if m["role"] == "user")


def test_extractor_marks_failure_after_all_retries_exhausted(fake_llm, usage_factory):
    for i in range(MAX_RETRIES + 1):
        fake_llm.queue(
            LLMValidationError(f"error #{i}", usage_factory(50, 5), "{}")
        )

    opp, retries, err, agg_usage = extract("Subject", "Body", fake_llm)

    assert opp is None
    assert retries == MAX_RETRIES
    assert err is not None
    assert "error #2" in err  # the last error is what surfaces
    # We made 3 attempts, so cost = 3 * (50, 5).
    assert agg_usage.prompt_tokens == 3 * 50
    assert agg_usage.completion_tokens == 3 * 5
    # Exactly MAX_RETRIES+1 calls were made.
    assert len(fake_llm.calls) == MAX_RETRIES + 1
