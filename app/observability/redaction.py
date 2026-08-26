"""
What leaves this process when tracing is on, and what must never leave.

What this module does and why it exists here: LangSmith traces are a copy of
the agent's working state uploaded to a third-party SaaS. That state is
recruiter email — real names, personal addresses, phone numbers, salary
figures — belonging to people who did not agree to have their mail sent to a
vendor so that someone could debug a graph. This module is the boundary. It
takes an arbitrary trace payload and returns one with the personal data
removed, and it is written so that the removal cannot be turned off by
configuration.

CONCEPT: two different redaction problems, two different tools.
  1. STRUCTURED fields — `recruiter_email`, `recruiter_phone`, `reply_to`.
     These have a known shape, so a regex genuinely removes them. An email
     address either matches the pattern or is not an email address.
  2. FREE TEXT — `body_text`, `jd_text`, `draft_body`. These contain names,
     employer names, team names and salary figures in ordinary prose. There
     is no regex for "is this token a person's name", and a redactor that
     claims to scrub prose mostly produces text that LOOKS clean while the
     names are still in it.
  So free text is DROPPED WHOLE, by path, and never pattern-matched. The
  honest version of "we can't sanitise this" is not sanitising it.

ALTERNATIVE considered and rejected: mask emails/phones inside body_text and
  keep the surrounding prose, which would make traces far more useful for
  debugging extraction (you could see WHY the model read a JD a certain way).
  Rejected for the live inbox because the residue — recruiter names, company
  names, CTC — is exactly the data D-for-publication anonymisation was about.
  If it is ever wanted, it belongs here as a second replacer selected by an
  explicit setting, NOT as a loosening of this one.

GOTCHA: this module must not import anything from app.pipeline or app.llm.
  It runs INSIDE the tracer's serialisation path. An import cycle here
  surfaces as a hang or a partial upload during tracing, which is a miserable
  thing to debug precisely because the debugging tool is what broke.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# What a redacted value is replaced with. A visible sentinel rather than an
# empty string: in a trace, "" reads as "the model got nothing here", which is
# a different bug from "we refused to upload this".
PLACEHOLDER = "[REDACTED]"
DROPPED = "[REDACTED:free-text]"

# CONCEPT: redaction by PATH, not by value.
#   The anonymizer walks the payload and hands us (value, path) for every
#   string, where path is the list of keys/indices that reach it — e.g.
#   ["inputs", "parsed", "body_text"]. Matching on the LAST key means we
#   catch `body_text` wherever it appears: at the top of node input, nested
#   under `opportunity`, inside a checkpoint blob, or in a list of messages.
#   Matching on the full path would need one entry per nesting site and would
#   silently miss any new one.
# WHY these specific keys: they are the free-text carriers on TriageState
#   (`parsed`), Opportunity (`jd_text`), and the drafting nodes
#   (`draft_body`). raw_headers is included because it holds the full
#   RFC 5322 header block, which contains the sender's display name.
FREE_TEXT_KEYS: frozenset[str] = frozenset({
    "body_text",
    "jd_text",
    "draft_body",
    "raw_headers",
    "subject",
    "from_name",
    "recruiter_name",
    "company",
    "end_client",
    # The LLM call payloads themselves. wrap_openai traces prompts and
    # completions under these keys, and the prompt is the email body.
    "content",
    "text",
})

# Structured patterns. Applied to every string we did NOT drop wholesale.
# GOTCHA: order matters. Email must run before phone, because an address like
# `asha.9876543210@corp.in` contains a phone-shaped run of digits; masking the
# digits first would leave a mangled address that still identifies the person.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Indian mobile numbers with optional +91 / 0 prefix and common separators,
# plus generic 10+ digit runs. Deliberately greedy: a false positive costs a
# masked number in a trace, a false negative costs someone's phone number.
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s\-.]?)?(?:\d[\s\-.]?){9,14}\d")
# Same PAN shape the outbound validator enforces (app/drafts/validator.py).
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_URL_RE = re.compile(r"https?://\S+")

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_EMAIL_RE, "[email]"),
    (_PAN_RE, "[pan]"),
    (_PHONE_RE, "[phone]"),
    (_URL_RE, "[url]"),
)


def _last_key(path: list[str | int]) -> str | None:
    """The nearest string key in `path`, skipping list indices.

    WHY skip indices: a body under `["messages", 0, "content"]` must match on
    `content`, not on `0`. List position tells us nothing about sensitivity.
    """
    for part in reversed(path):
        if isinstance(part, str):
            return part
    return None


def mask_patterns(value: str) -> str:
    """Mask structured identifiers inside a string that we are keeping."""
    for pattern, replacement in _PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def redact(value: str, path: list[str | int]) -> str:
    """The replacer handed to LangSmith: one string in, one safe string out.

    TRACE: called once per string in the trace payload, on the upload path,
    before anything crosses the network. Both trace sources — the LangGraph
    node tracer and the wrapped Azure client — go through this same function,
    because both are given the same Client instance (see tracing.py).

    GOTCHA: this function must never raise. See `safe_redact` — it is the one
    that is actually installed, and this is the body it guards.
    """
    key = _last_key(path)
    if key is not None and key in FREE_TEXT_KEYS:
        # No pattern matching attempted. See the module CONCEPT: prose is not
        # sanitisable, so it does not travel.
        return DROPPED
    return mask_patterns(value)


def safe_redact(value: str, path: list[str | int]) -> str:
    """`redact`, but a failure yields the placeholder instead of the original.

    CONCEPT: fail-closed on the redactor itself.
      Every other guard in this system fails OPEN, and deliberately so —
      `already_replied` returning False on a DB error means we send a possibly
      duplicate reply rather than silently drop a real recruiter (see D60).
      That reasoning inverts here. The cost of this function failing open is
      an unredacted email body on a vendor's servers, which cannot be undone
      by noticing it later. The cost of failing closed is a less useful trace.
      So: any exception, any non-string, any surprise — emit the placeholder.

    GOTCHA: we log at WARNING and do NOT re-raise. Raising inside the tracer
    would take down an ingest cycle for the sake of observability, which
    inverts what observability is for.
    """
    try:
        if not isinstance(value, str):
            return PLACEHOLDER
        return redact(value, path)
    except Exception as exc:
        log.warning(
            "redaction failed at path %r (%s: %s); emitting placeholder",
            path, type(exc).__name__, exc,
        )
        return PLACEHOLDER


def build_anonymizer():
    """Return the callable LangSmith applies to every trace payload.

    `create_anonymizer` accepts a replacer of shape
    `(value: str, path: list[str | int]) -> str` and returns
    `Callable[[Any], Any]` that walks an arbitrary payload applying it.

    WHY build it here rather than at import: `langsmith` is a transitive
    dependency of langchain-core, and importing it at module scope would make
    this module — which the tests exercise directly — fail to import in an
    environment that has trimmed it. The pure functions above stay testable
    with no langsmith installed at all.
    """
    from langsmith.anonymizer import create_anonymizer

    return create_anonymizer(safe_redact)


def redact_payload(payload: Any) -> Any:
    """Convenience wrapper used by the tests to check whole-payload behaviour."""
    return build_anonymizer()(payload)
