"""
Outbound validator tests. Pure — no LLM, no DB.

The load-bearing property: rejection is quarantine + reason, never silent
strip. If any of these tests start passing by returning `quarantined=False`
with a mutated body, the safety promise of the module has broken.
"""

from __future__ import annotations

import pytest

from app.drafts.validator import validate_draft


# --- PAN ---


def test_pan_at_end_quarantines():
    body = "Please share your CV to ABCDE1234F"
    v = validate_draft(body, max_length_chars=1000)
    assert v.quarantined
    assert "PAN" in v.reason
    assert v.body_text == body  # UNCHANGED — never silently strip.


def test_pan_embedded_in_paragraph_quarantines():
    body = "Hi Rajesh, my PAN is ABCDE1234F for your records. Thanks."
    v = validate_draft(body, 1000)
    assert v.quarantined


def test_pan_lowercase_does_not_match():
    """PAN is uppercase-only per the format definition."""
    body = "abcde1234f is not a PAN by the spec"
    v = validate_draft(body, 1000)
    assert not v.quarantined


# --- Aadhaar ---


@pytest.mark.parametrize(
    "aadhaar",
    ["123456789012", "1234 5678 9012", "1234-5678-9012"],
)
def test_aadhaar_variants_quarantine(aadhaar):
    body = f"My Aadhaar is {aadhaar} for reference"
    v = validate_draft(body, 1000)
    assert v.quarantined
    assert "Aadhaar" in v.reason


def test_aadhaar_shape_does_not_match_11_digits():
    """11 digits is not Aadhaar shape."""
    body = "Number 12345678901 is only 11 digits"
    v = validate_draft(body, 1000)
    assert not v.quarantined


def test_aadhaar_word_boundary_prevents_mid_integer_match():
    """A 20-digit blob should not fire on a 12-digit substring."""
    body = "Reference 12345678901234567890 for tracking"
    v = validate_draft(body, 1000)
    assert not v.quarantined


# --- Length ---


def test_over_max_length_quarantines():
    body = "x" * 5000
    v = validate_draft(body, max_length_chars=3000)
    assert v.quarantined
    assert "length" in v.reason
    assert v.body_text == body  # unchanged


def test_at_max_length_passes():
    body = "x" * 3000
    v = validate_draft(body, max_length_chars=3000)
    assert not v.quarantined


# --- Clean ---


def test_clean_draft_passes():
    body = (
        "Hi Priya,\n\nThanks for reaching out about the .NET role at Acme. "
        "This one isn't a fit right now — the CTC ceiling is below what I'm "
        "currently at.\n\nPlease keep me in mind for future roles."
    )
    v = validate_draft(body, max_length_chars=3000)
    assert not v.quarantined
    assert v.reason is None
    assert v.body_text == body


# --- Never mutates ---


def test_body_text_is_never_mutated_on_reject():
    """Quarantine must return the original bytes, not a scrubbed version."""
    body = "PAN ABCDE1234F is here"
    v = validate_draft(body, 1000)
    assert v.body_text == body
    assert "ABCDE1234F" in v.body_text  # NOT masked, NOT redacted
