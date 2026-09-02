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

CONCEPT: the second replacer, which is now real (D75).
  The paragraph above prescribed the shape and this module now implements it.
  `identifiers=True` — driven by LANGSMITH_TRACE_IDENTIFIERS, never by
  anything else — moves exactly three fields (subject, from_name, from_email)
  out of the dropped set and passes them through UNMASKED. It does not widen
  the prose rule by one byte: body_text, jd_text, draft_body, raw_headers,
  content and text are dropped whole in both modes.
  WHY unmasked rather than pattern-masked for those three: the entire purpose
  is answering "which email is this, and who sent it". `_EMAIL_RE` would
  rewrite from_email to `[email]`, which is precisely the information being
  asked for. A half-measure here would cost the privacy and not buy the
  monitoring, which is the worst of both.
  GOTCHA: `identifiers` is a keyword-only argument with a False default on
  every function here. That is deliberate — a caller that forgets it gets
  STRICT behaviour, so the failure mode of a future refactor is a less useful
  trace rather than an unredacted one.

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

# The three fields that identify a message without describing it, released
# only when `identifiers=True`. Anything not on this list stays governed by
# FREE_TEXT_KEYS above — note `recruiter_name`, `company` and `end_client` are
# deliberately NOT here. Those are EXTRACTED values, and the point of a trace
# you are auditing for correctness is to check the extraction against a source
# you already trust; the From header and the Subject line are that source.
# WHY the metadata spellings sit beside the state ones: app/observability/
# tracing.py stamps the same three values into run metadata under `email_*`
# names, and LangSmith does not run metadata through the anonymizer at all.
# Listing both spellings means the two channels cannot drift into disagreeing
# about what is releasable — one list, one policy.
IDENTIFIER_KEYS: frozenset[str] = frozenset({
    "subject",
    "from_name",
    "from_email",
    "email_subject",
    "email_from_name",
    "email_from",
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


def _free_text_ancestor(path: list[str | int]) -> bool:
    """True if a key ABOVE the leaf names a free-text carrier.

    WHY the leaf is excluded rather than included: the leaf's own key is
    handled separately, further down `redact`, and it has to be — `subject` is
    a member of BOTH FREE_TEXT_KEYS and IDENTIFIER_KEYS, so a check that saw
    its own key here would drop it before the identifier allowlist was
    consulted and the setting would appear inert. An ANCESTOR, by contrast,
    admits no exception: a string sitting inside `raw_headers` or inside an
    LLM `content` block is part of that prose no matter what it is called, so
    it drops before anything else is asked.

    GOTCHA this exists to fix — a real leak, found 2026-09-02.
      Matching on the LAST key alone is correct only when the free-text field
      holds a string. `raw_headers` holds a DICT, so the anonymizer descends
      into it and hands us the path ["parsed", "raw_headers", "From"] with
      "From" as the last key. "From" is not in FREE_TEXT_KEYS, so the value
      fell through to mask_patterns, which strips the address and keeps the
      display name — publishing "Brightpath Careers <[email]>" to LangSmith
      out of a field listed in FREE_TEXT_KEYS *specifically because* the
      module comment says it "contains the sender's display name".
      Checking every key on the path means a free-text field drops its whole
      SUBTREE, not just its own string, so nesting can no longer route around
      the rule. The same hole would have opened the first time any message
      list, tool-call block or structured content field arrived as a nested
      object rather than a flat string.
    """
    seen_leaf = False
    for part in reversed(path):
        if not isinstance(part, str):
            continue
        if not seen_leaf:
            # The nearest string key is the leaf's own name — skip it.
            seen_leaf = True
            continue
        if part in FREE_TEXT_KEYS:
            return True
    return False


def mask_patterns(value: str) -> str:
    """Mask structured identifiers inside a string that we are keeping."""
    for pattern, replacement in _PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def redact(value: str, path: list[str | int], *, identifiers: bool = False) -> str:
    """The replacer handed to LangSmith: one string in, one safe string out.

    TRACE: called once per string in the trace payload, on the upload path,
    before anything crosses the network. Both trace sources — the LangGraph
    node tracer and the wrapped Azure client — go through this same function,
    because both are given the same Client instance (see tracing.py).

    GOTCHA: this function must never raise. See `safe_redact` — it is the one
    that is actually installed, and this is the body it guards.
    """
    # The three checks below are ordered, and the order is the whole logic:
    #   1. Anything INSIDE a prose field drops. No exception, either mode.
    #   2. Otherwise, an allowlisted identifier is released — but only when
    #      the setting is on. `subject` and `from_name` are members of both
    #      sets, so testing FREE_TEXT_KEYS before this would drop them before
    #      the allowlist was ever consulted and the setting would appear
    #      inert — a bug that reads as "tracing is broken" rather than as an
    #      ordering mistake.
    #   3. Otherwise the ordinary rule: prose drops, everything else is
    #      pattern-masked.
    key = _last_key(path)
    if _free_text_ancestor(path):
        return DROPPED
    if identifiers and key is not None and key in IDENTIFIER_KEYS:
        return value
    if key is not None and key in FREE_TEXT_KEYS:
        # No pattern matching attempted. See the module CONCEPT: prose is not
        # sanitisable, so it does not travel.
        return DROPPED
    return mask_patterns(value)


def safe_redact(
    value: str, path: list[str | int], *, identifiers: bool = False
) -> str:
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
        return redact(value, path, identifiers=identifiers)
    except Exception as exc:
        log.warning(
            "redaction failed at path %r (%s: %s); emitting placeholder",
            path, type(exc).__name__, exc,
        )
        return PLACEHOLDER


def build_anonymizer(*, identifiers: bool = False):
    """Return the callable LangSmith applies to every trace payload.

    `create_anonymizer` accepts a replacer of shape
    `(value: str, path: list[str | int]) -> str` and returns
    `Callable[[Any], Any]` that walks an arbitrary payload applying it.

    WHY build it here rather than at import: `langsmith` is a transitive
    dependency of langchain-core, and importing it at module scope would make
    this module — which the tests exercise directly — fail to import in an
    environment that has trimmed it. The pure functions above stay testable
    with no langsmith installed at all.

    GOTCHA: `identifiers` is bound into the closure HERE, at client-build
    time, and the client is built once per process (tracing.get_client is
    lru_cached). Flipping the setting therefore needs a restart, exactly like
    LANGSMITH_TRACING itself. Anything that reads the flag per-call would
    imply it could change mid-run, which it cannot.
    """
    from langsmith.anonymizer import create_anonymizer

    def replacer(value: Any, path: list[str | int]) -> str:
        return safe_redact(value, path, identifiers=identifiers)

    return create_anonymizer(replacer)


def redact_payload(payload: Any, *, identifiers: bool = False) -> Any:
    """Convenience wrapper used by the tests to check whole-payload behaviour.

    GOTCHA: the anonymizer MUTATES `payload` in place and returns it — it is
    not a pure function, whatever the return value suggests. Redacting the
    same dict twice therefore redacts the already-redacted copy, which reads
    as "the identifier release did nothing" and sends you looking for a bug in
    `redact`. Build a fresh payload per call. This is harmless in production,
    where LangSmith hands it a serialised copy it owns, and a trap in any
    script that loops over modes to compare them.
    """
    return build_anonymizer(identifiers=identifiers)(payload)
