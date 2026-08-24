"""
Regression tests for D44 — NUL bytes in untrusted email content.

A recruiter mail carrying 0x00 killed a 100-message ingest at message 42:
the byte survived parse and extraction, and PostgreSQL rejected the INSERT
with `DataError: text fields cannot contain NUL (0x00) bytes`. These tests
pin the sanitisation at the boundary where the content enters.
"""

from __future__ import annotations

import base64

from app.gmail.parser import _decode_body_data, _headers_dict, scrub_text


def _b64(s: str) -> dict:
    """Encode like Gmail does: base64url with padding stripped."""
    return {"data": base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")}


def test_scrub_removes_nul_preserves_everything_else():
    assert scrub_text("a\x00b") == "ab"
    assert scrub_text("\x00leading") == "leading"
    assert scrub_text("trailing\x00") == "trailing"
    assert scrub_text("many\x00\x00nuls") == "manynuls"


def test_scrub_is_identity_for_clean_text():
    # WHY assert identity and not just equality: scrub_text short-circuits
    # when there is no NUL, so the common path must not copy the string.
    s = "Perfectly ordinary recruiter mail — 30 LPA, hybrid, Pune."
    assert scrub_text(s) is s


def test_scrub_keeps_other_control_and_unicode_characters():
    # Only NUL is illegal in a PG text column. Newlines and tabs are load-
    # bearing in JD text (bullet lists), and unicode is everywhere in Indian
    # recruiter mail (₹, →, em dashes). Stripping those would be a regression.
    s = "line1\nline2\ttabbed ₹30L → ok"
    assert scrub_text(s) == s


def test_decoded_body_is_scrubbed():
    """The exact production path: base64 body containing a NUL."""
    body = _b64("We are looking for .NET Leads\x00 with 8+ years.")
    out = _decode_body_data(body)
    assert "\x00" not in out
    assert out.startswith("We are looking for .NET Leads")


def test_headers_are_scrubbed():
    """Subject and From land in TEXT columns too, so they get the same pass."""
    headers = _headers_dict(
        [
            {"name": "Subject", "value": "Urgent\x00 hiring: .NET Lead"},
            {"name": "From", "value": "Arvind K <arvind@example.com>"},
        ]
    )
    assert headers["Subject"] == "Urgent hiring: .NET Lead"
    assert headers["From"] == "Arvind K <arvind@example.com>"


def test_empty_body_data_returns_empty_string():
    # Guards the early-return branch — a part with no `data` key at all.
    assert _decode_body_data({}) == ""
