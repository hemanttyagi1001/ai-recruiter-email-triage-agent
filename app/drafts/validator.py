"""
Outbound validator — the last gate before a draft is persisted.

CONCEPT: Why a rule that must never be violated cannot live in a prompt.
  A prompt is a stochastic instruction to a probabilistic system. "Never
  include a PAN in the reply" is a *request*. Sample the same prompt across
  10,000 messages and one output will slip through anyway — different
  temperature, longer context, model update, prompt injection in the JD.
  You cannot bound the failure rate with a prompt.

  A code validator is a deterministic constraint: a regex either matches
  or it doesn't. Zero variance. The right pattern is layered:
    - prompt: "don't include PII" (reduces the base rate of attempts)
    - code:   "no PII shall pass" (drives the covered patterns to zero)

  This module is the "shall pass" side. It runs after the draft exists,
  outside the LLM's reach. The LLM cannot argue with it, cannot claim
  "actually this specific PAN is fine because," cannot be tricked by
  injected text — the injected text is precisely what the validator scans.

  Rejection is QUARANTINE, not silent strip. Never edit the model's output
  and pass it forward — that hides the failure and makes it harder to
  audit. Store the raw draft with a quarantine reason; the human review
  step decides what to do.

Regexes:
  - PAN (Indian Permanent Account Number): five uppercase letters, four
    digits, one uppercase letter. Word-boundary anchored.
  - Aadhaar (India national id): 12 digits, optionally split by single
    space or hyphen after each 4-digit block. Word-boundary anchored so we
    don't fire on random 12-digit substrings inside longer numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# PAN: exactly 5 letters + 4 digits + 1 letter, at word boundaries.
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Aadhaar: 12 digits, optionally separated by a single space or hyphen
# every 4 digits. \b anchors ensure we don't fire mid-integer.
AADHAAR_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")


@dataclass(frozen=True)
class ValidationVerdict:
    quarantined: bool
    reason: str | None
    # Body text passes through untouched, always. Callers get the same
    # string they sent in — this class does not sanitise.
    body_text: str

    @classmethod
    def clean(cls, body: str) -> "ValidationVerdict":
        return cls(quarantined=False, reason=None, body_text=body)

    @classmethod
    def quarantine(cls, body: str, reason: str) -> "ValidationVerdict":
        return cls(quarantined=True, reason=reason, body_text=body)


def validate_draft(body: str, max_length_chars: int) -> ValidationVerdict:
    """Run all outbound checks. First-match-wins on the reason string.

    Order matters only for the reason returned when a draft violates
    multiple rules; the outcome (quarantined) is identical.
    """
    m = PAN_PATTERN.search(body)
    if m:
        return ValidationVerdict.quarantine(
            body, reason=f"contains PAN-shaped token {m.group(0)!r}"
        )

    m = AADHAAR_PATTERN.search(body)
    if m:
        return ValidationVerdict.quarantine(
            body, reason=f"contains Aadhaar-shaped token {m.group(0)!r}"
        )

    if len(body) > max_length_chars:
        return ValidationVerdict.quarantine(
            body,
            reason=f"length {len(body)} exceeds configured max {max_length_chars}",
        )

    return ValidationVerdict.clean(body)
