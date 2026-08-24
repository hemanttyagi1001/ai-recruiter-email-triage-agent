"""
Outbound MIME construction (D52).

The HTML part exists only to carry typography. Everything here guards the two
ways that can go wrong: the styled part diverging from the validated plain
text, and untrusted extractor output becoming live markup in mail we send to
somebody else.
"""

from __future__ import annotations

import pytest

from app.gmail.client import _css_font_stack, build_mime, encode_mime


def _parts(msg):
    return {p.get_content_type(): p.get_payload(decode=True).decode() for p in msg.get_payload()}


BODY = "Hi Ritika,\n\n  - Total experience: 14 years\n\nWarm Regards\nHemant"


def test_message_is_multipart_alternative_plain_then_html():
    """Order matters: RFC 2046 says the LAST part is the preferred one.

    Attaching HTML first would make most clients render the unstyled text and
    the font setting would silently do nothing.
    """
    msg = build_mime("hr@x.com", "Re: Role", BODY)
    assert msg.get_content_type() == "multipart/alternative"
    types = [p.get_content_type() for p in msg.get_payload()]
    assert types == ["text/plain", "text/html"]


def test_plain_part_is_the_body_verbatim():
    """The plain part must be exactly what the validator inspected.

    If this ever diverges, `drafts.body_text` in the database stops being a
    record of what was actually sent.
    """
    msg = build_mime("hr@x.com", "Re: Role", BODY)
    assert _parts(msg)["text/plain"] == BODY


def test_html_part_carries_the_configured_font():
    msg = build_mime("hr@x.com", "Re: Role", BODY, font_family="Trebuchet MS")
    html = _parts(msg)["text/html"]
    assert "'Trebuchet MS'" in html
    # A fallback stack matters — recipients without the font should not land
    # on a serif default that looks nothing like the intended mail.
    assert "sans-serif" in html


def test_html_preserves_indentation_and_line_breaks():
    """pre-wrap, not <br>: the templates indent bullet lists with two spaces.

    HTML collapses leading whitespace, so without pre-wrap the "  - Total
    experience" lines would lose their indent and read as a wall of text.
    """
    html = _parts(build_mime("hr@x.com", "s", BODY))["text/html"]
    assert "white-space: pre-wrap" in html
    assert "  - Total experience: 14 years" in html


# --- the security case -------------------------------------------------------


@pytest.mark.parametrize("payload", [
    "<script>alert(1)</script>",
    '<img src=x onerror="steal()">',
    '<a href="http://evil">click</a>',
    "</div><style>body{display:none}</style>",
])
def test_untrusted_body_content_is_escaped_not_rendered(payload):
    """A crafted role title must not become markup in mail WE send.

    body_text embeds extractor output, and the extractor reads attacker-
    controlled recruiter email. Without escaping, this system would forward
    hostile markup to a third party under the user's own name — turning the
    agent into a delivery vehicle for exactly the content it exists to
    quarantine.
    """
    body = f"Thanks for reaching out about the {payload} role."
    parts = _parts(build_mime("hr@x.com", "s", body))
    assert payload not in parts["text/html"], "payload survived unescaped"
    assert "&lt;" in parts["text/html"]
    # The plain part is untouched — escaping is a rendering concern only.
    assert payload in parts["text/plain"]


def test_font_name_cannot_break_out_of_the_style_attribute():
    """The font comes from config, not from email, but it still gets sanitised.

    A stray quote would close the style attribute and put arbitrary text into
    the tag. Cheap to prevent; confusing to debug if it ever happened.
    """
    stack = _css_font_stack('X"; background:url(evil); font-family:"Y')
    assert '"' not in stack
    assert ";" not in stack
    assert "(" not in stack


def test_empty_font_falls_back_rather_than_emitting_broken_css():
    assert "Trebuchet MS" in _css_font_stack("")
    assert "Trebuchet MS" in _css_font_stack("!!!")


# --- headers -----------------------------------------------------------------


def test_threading_headers_set_together():
    msg = build_mime("hr@x.com", "Re: Role", BODY, in_reply_to_message_id="<abc@x>")
    assert msg["In-Reply-To"] == "<abc@x>"
    assert msg["References"] == "<abc@x>"


def test_threading_headers_absent_when_not_a_reply():
    msg = build_mime("hr@x.com", "Role", BODY)
    assert msg["In-Reply-To"] is None
    assert msg["References"] is None


def test_encode_mime_is_url_safe_base64():
    """Gmail's `raw` field is base64URL — '+' and '/' would corrupt it."""
    raw = encode_mime(build_mime("hr@x.com", "s", BODY))
    assert "+" not in raw and "/" not in raw


def test_unicode_body_survives_the_round_trip():
    """Signatures carry emoji and Indian recruiter mail carries ₹ and em dashes."""
    body = "Hi there,\n\n₹30L — remote\n📞 +91 9548550009"
    parts = _parts(build_mime("hr@x.com", "s", body))
    assert parts["text/plain"] == body
    assert "₹30L" in parts["text/html"]
    assert "📞" in parts["text/html"]
