"""
Tests for the tracing redaction boundary.

These are the tests that decide whether it is safe to turn tracing on at all,
so they assert the SAFE direction: not "the redactor did something" but "no
recognisable personal datum survives anywhere in the payload". A test that
checks `body_text == "[REDACTED:free-text]"` passes just as happily when a
copy of the body is still sitting under some other key, which is why
`test_no_email_survives_anywhere_in_a_realistic_payload` walks the whole
structure instead of naming fields.

All offline. Nothing here needs a LangSmith key, a network, or a database.
"""

from __future__ import annotations

import json

import pytest

from app.observability.redaction import (
    DROPPED,
    PLACEHOLDER,
    mask_patterns,
    redact,
    redact_payload,
    safe_redact,
)


# --- Structured masking -----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "asha@midwestconsultants.net",
        "sunita.rao02@northwind.com",
        "careers@brightpathtech.in",
        "first.last+tag@sub.domain.co.in",
    ],
)
def test_email_addresses_are_masked(raw):
    assert "@" not in mask_patterns(f"reach me at {raw} today")


@pytest.mark.parametrize(
    "raw",
    ["+91 98765 43210", "9876543210", "+91-98765-43210", "098765 43210"],
)
def test_phone_numbers_are_masked(raw):
    out = mask_patterns(f"call {raw}")
    assert "98765" not in out


def test_pan_is_masked():
    # Same shape the outbound validator quarantines on.
    assert "ABCDE1234F" not in mask_patterns("PAN ABCDE1234F on file")


def test_email_containing_digits_does_not_leak_a_partial_address():
    """GOTCHA regression: phone-first ordering mangles rather than removes.

    `asha.9876543210@corp.in` has a phone-shaped digit run inside it. If the
    phone pattern ran first the result would be `asha.[phone]@corp.in` — still
    naming the person and their employer.
    """
    out = mask_patterns("write to asha.9876543210@corp.in")
    assert "asha" not in out
    assert "corp.in" not in out


# --- Free text is dropped, never pattern-matched ----------------------------


@pytest.mark.parametrize(
    "key",
    ["body_text", "jd_text", "draft_body", "raw_headers", "subject",
     "recruiter_name", "company", "content"],
)
def test_free_text_fields_are_dropped_wholesale(key):
    prose = "Hi Hemant, Priya here from Brightpath about a 32 LPA role."
    assert redact(prose, ["inputs", key]) == DROPPED


def test_free_text_is_dropped_at_any_nesting_depth():
    """Matching is on the nearest string key, so list indices don't hide it."""
    prose = "Priya from Brightpath, 32 LPA"
    assert redact(prose, ["inputs", "messages", 0, "content"]) == DROPPED
    assert redact(prose, ["outputs", "opportunity", "jd_text"]) == DROPPED


def test_non_free_text_fields_are_masked_not_dropped():
    """Structured fields keep their shape so traces stay debuggable."""
    out = redact("asha@corp.in", ["outputs", "reply_to"])
    assert out != DROPPED
    assert "@" not in out


# --- Fail-closed ------------------------------------------------------------


def test_replacer_never_raises_and_never_returns_the_original(monkeypatch):
    """A redactor that throws must not fall back to the unredacted value.

    GOTCHA the first version of this test walked into: faking the failure with
    a str subclass whose `__len__` raises proves nothing, because `re.sub`
    never calls `__len__` — the redaction simply succeeded and the test failed
    asserting the wrong constant. The only honest way to test a guard is to
    break the thing it guards, so `redact` itself is replaced here.
    """
    import app.observability.redaction as redaction

    def boom(value, path):
        raise RuntimeError("redactor blew up mid-walk")

    monkeypatch.setattr(redaction, "redact", boom)
    out = redaction.safe_redact("priya@corp.in", ["inputs", "reply_to"])
    assert out == PLACEHOLDER
    assert "priya" not in out


def test_a_broken_redactor_leaks_nothing_from_a_whole_payload(monkeypatch):
    """Fail-closed at payload scale, not just for one string."""
    import app.observability.redaction as redaction

    monkeypatch.setattr(
        redaction, "redact", lambda value, path: (_ for _ in ()).throw(RuntimeError())
    )
    dumped = json.dumps(redaction.redact_payload(_realistic_payload()))
    for leak in ("priya", "brightpathtech", "98765", "Sharma"):
        assert leak not in dumped


def test_non_string_values_yield_the_placeholder():
    assert safe_redact(object(), ["inputs", "x"]) == PLACEHOLDER


# --- Whole-payload behaviour ------------------------------------------------


def _realistic_payload() -> dict:
    """Shaped like a real TriageState node input, PII in every carrier."""
    return {
        "inputs": {
            "parsed": {
                "message_id": "<mid-1@brightpathtech.in>",
                "from_email": "careers@brightpathtech.in",
                "from_name": "Brightpath Careers",
                "subject": "Opening for Senior AI/ML Engineer",
                "body_text": (
                    "Hi Hemant, this is Priya Sharma from Brightpath. We have a "
                    "32 LPA role. Call me on +91 98765 43210 or write to "
                    "priya.sharma@brightpathtech.in."
                ),
                "raw_headers": {"From": "Brightpath Careers <careers@brightpathtech.in>"},
            },
            "opportunity": {
                "company": "Brightpath Tech",
                "recruiter_name": "Priya Sharma",
                "recruiter_email": "priya.sharma@brightpathtech.in",
                "recruiter_phone": "+91 98765 43210",
                "jd_text": "Senior AI/ML Engineer, Bangalore, 32 LPA",
                "ctc_max_lpa": 32,
            },
            "messages": [
                {"role": "user", "content": "Extract from: Priya Sharma, Brightpath, 32 LPA"}
            ],
        },
        "outputs": {
            "reply_to": "priya.sharma@brightpathtech.in",
            "draft_body": "Hi Priya, thanks for reaching out about the role...",
        },
    }


@pytest.mark.parametrize(
    "leak",
    [
        "priya.sharma@brightpathtech.in",
        "careers@brightpathtech.in",
        "Priya Sharma",
        "98765",
        "Brightpath Tech",
    ],
)
def test_no_pii_survives_anywhere_in_a_realistic_payload(leak):
    """The load-bearing test: serialise the redacted payload and grep it.

    WHY assert over the serialised form rather than field by field: a
    field-by-field assertion cannot notice a copy of the body that ended up
    somewhere the test did not think to look. This is the check that would
    catch a future key carrying prose that FREE_TEXT_KEYS does not list.
    """
    redacted = redact_payload(_realistic_payload())
    assert leak not in json.dumps(redacted)


def test_redaction_preserves_structure_and_non_pii_scalars():
    """Strict redaction must not flatten the run tree — that's what you debug with."""
    redacted = redact_payload(_realistic_payload())
    assert set(redacted) == {"inputs", "outputs"}
    assert "opportunity" in redacted["inputs"]
    # Numbers are not strings; the replacer never sees them, so the shape of
    # the state (and anything numeric a router branched on) still reads.
    assert redacted["inputs"]["opportunity"]["ctc_max_lpa"] == 32
