"""
Turns a Gmail v1 message JSON into a typed `ParsedMessage` — the shape the
rest of the pipeline consumes. The interesting work here is MIME-tree walking
and body extraction. Recruiter email is mixed text/HTML, quoted replies,
occasional multipart/alternative wrappers; this parser normalizes all of it
to a single plain-text body.

Header extraction produces the RFC 5322 Message-ID that becomes our
idempotency key (see the CONCEPT comment on Message.message_id in
db/models.py). If a message lacks a Message-ID header (rare — old MTAs,
broken clients), we synthesise one from the Gmail id so the DB column is
never null.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime

from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


@dataclass
class ParsedMessage:
    """Pipeline-facing shape of a Gmail message."""

    message_id: str  # RFC 5322 Message-ID header (with angle brackets kept)
    gmail_id: str
    thread_id: str
    from_email: str
    from_name: str | None
    subject: str
    received_at: datetime
    body_text: str
    raw_headers: dict[str, str]


def parse_message(raw: dict) -> ParsedMessage:
    """Convert a Gmail `format=full` payload to a `ParsedMessage`."""
    payload = raw.get("payload", {})
    headers = _headers_dict(payload.get("headers", []))
    # Header lookups are case-insensitive per RFC 5322.
    lower = {k.lower(): v for k, v in headers.items()}

    from_raw = lower.get("from", "")
    from_name_raw, from_email = parseaddr(from_raw)
    from_name = from_name_raw or None

    date_raw = lower.get("date")
    received_at = _parse_date(date_raw)

    subject = lower.get("subject", "(no subject)")

    # GOTCHA: Message-ID can be missing in the wild. Rather than fail the run
    # or store null, synthesise a locally-unique id from the Gmail id. It's
    # not portable across mail providers but neither is a message without an
    # RFC-compliant Message-ID — the invariant that gets preserved is
    # "column is never null, PK collisions still work within Gmail."
    message_id = lower.get("message-id") or f"<gmail-{raw['id']}@synth.local>"

    body_text = _extract_body(payload)

    return ParsedMessage(
        message_id=message_id,
        gmail_id=raw["id"],
        thread_id=raw["threadId"],
        from_email=from_email,
        from_name=from_name,
        subject=subject,
        received_at=received_at,
        body_text=body_text,
        raw_headers=headers,
    )


def scrub_text(s: str) -> str:
    """Remove characters PostgreSQL TEXT columns cannot store.

    CONCEPT: sanitise untrusted input at the boundary where it enters, not at
    the boundary where it breaks. Email bodies are attacker-influenced and
    arrive from arbitrary MIME encoders; a NUL (0x00) byte is legal in a
    Python str and illegal in a PostgreSQL text field. Left alone it travels
    all the way through classify, extract, embed and scoring before psycopg
    rejects the INSERT — by which point several LLM calls have been paid for
    and the exception surfaces from a persist node with no obvious link to
    the malformed input.

    GOTCHA: this is exactly what happened on 2026-08-21. A `.NET Lead` mail
    from Naukri carried a NUL, the extractor faithfully copied it into
    jd_text, and `DataError: PostgreSQL text fields cannot contain NUL (0x00)
    bytes` killed an entire 100-message run at message 42. See D44.

    WHY strip rather than reject the message: a NUL is almost always an
    artefact of a broken encoder, not a signal about content. The surrounding
    text is still perfectly good recruiter mail, and discarding a real
    opportunity over one stray byte is the worse failure.
    """
    return s.replace("\x00", "") if "\x00" in s else s


def _headers_dict(entries: list[dict]) -> dict[str, str]:
    """Gmail returns headers as [{'name': .., 'value': ..}, ...]. Same header
    can appear multiple times (Received, DKIM-Signature). We keep the last —
    good enough for our fields (From/Subject/Date/Message-ID are singletons).

    WHY scrubbed here too: subject and from_name land in TEXT columns on
    `messages`. The observed NUL was in a body, but nothing stops one
    appearing in a header, and the failure would look identical.
    """
    return {h["name"]: scrub_text(h["value"]) for h in entries}


def _parse_date(date_raw: str | None) -> datetime:
    if not date_raw:
        return datetime.now(timezone.utc)
    try:
        dt = parsedate_to_datetime(date_raw)
        # WHY the utcoffset check: parsedate_to_datetime returns naive
        # datetime when the header lacks a timezone (RFC 5322 says it MUST
        # have one but real mail cheats). Naive datetime through timezone-
        # aware DB columns is a data-integrity hazard — force UTC.
        if dt.utcoffset() is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        log.warning("Unparseable Date header %r; using now()", date_raw)
        return datetime.now(timezone.utc)


def _extract_body(payload: dict) -> str:
    """Prefer text/plain; fall back to HTML→text via BeautifulSoup.

    See DECISIONS.md D4 for why. In short: text/plain preserves recruiter-
    authored formatting when present; HTML fallback catches the ~40% of
    recruiter templates that ship HTML-only.
    """
    plain = _find_part(payload, "text/plain")
    if plain is not None:
        return _decode_body_data(plain.get("body", {}))

    html = _find_part(payload, "text/html")
    if html is not None:
        raw_html = _decode_body_data(html.get("body", {}))
        # get_text with a separator collapses <br>/<p> boundaries to newlines,
        # keeping paragraph structure without markup noise.
        return BeautifulSoup(raw_html, "html.parser").get_text("\n", strip=True)

    # Some tiny messages have the body directly on payload (no parts).
    body_data = payload.get("body", {}).get("data")
    if body_data:
        return _decode_body_data(payload["body"])

    return ""


def _find_part(payload: dict, mime_type: str) -> dict | None:
    """Depth-first search for the first part matching `mime_type`."""
    # TRACE: multipart/alternative → [text/plain, text/html]. We recurse.
    # multipart/mixed → [multipart/alternative, application/pdf]. We recurse
    # past the multipart wrapper. Attachments (application/*) are ignored —
    # Phase 0 does not process attachments.
    if payload.get("mimeType") == mime_type:
        return payload
    for part in payload.get("parts", []) or []:
        found = _find_part(part, mime_type)
        if found is not None:
            return found
    return None


def _decode_body_data(body: dict) -> str:
    """Gmail body data is base64url with stripped padding. Restore and decode."""
    data = body.get("data", "")
    if not data:
        return ""
    # GOTCHA: urlsafe_b64decode requires padding to be a multiple of 4.
    # Gmail strips trailing '='. Adding three '=' is always safe (Python's
    # decoder ignores excess padding).
    # TRACE: every body path — text/plain, the HTML fallback, and the
    # single-part payload case — funnels through this one function, so
    # scrubbing here covers all of them with one call. See scrub_text.
    decoded = base64.urlsafe_b64decode(data + "===").decode("utf-8", errors="replace")
    return scrub_text(decoded)
