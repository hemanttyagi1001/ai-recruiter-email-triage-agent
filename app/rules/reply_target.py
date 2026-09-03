"""
Who, if anyone, should a reply actually go to.

What this module does and why it exists here: more than half of inbound
"recruiter" mail is not from a recruiter. It is a job board's alert digest,
sent from an address that discards anything you send back. Replying to those
costs tokens, clutters the approval queue, and — once autonomous sending is
armed — emits mail nobody will ever read. This module answers one question
deterministically: given a parsed message and whatever the extractor found,
what human address should receive the reply, or is there none?

CONCEPT: three shapes of sender, only two of which are worth answering.
  A. DIRECT — a person at a company wrote to you. `ritika.s@brightpathtech.in`.
     Reply to the sender. Easy case.
  B. RELAY — a real recruiter wrote to you *through* a portal, which rewrote
     the From header into an opaque forwarding address. Naukri does this by
     base64-encoding the recruiter's domain into the local part:
         ashabWlkd2VzdGNvbnN1bHRhbnRzLm5ldA==@naukri.com
         base64("midwestconsultants.net")
     Their real address usually appears in the body ("share your CV to
     asha@midwestconsultants.net"), and the extractor already captures
     it as `Opportunity.recruiter_email`. So the human IS reachable — just not
     at the address in the From header.
  C. ALERT — a machine generated a digest. `noreply@glassdoor.com`,
     `jobalerts-noreply@linkedin.com`, `naukrialerts@naukri.com`. Nobody is
     there. No reply should be drafted at all.

WHY this is code and not a prompt: it decides whether an email gets sent. The
project's standing rule is that a constraint which must not be violated lives
in a validator, not in a string an LLM might reinterpret — and unlike a
judgement call about fit, "does this address accept mail" is a fact a lookup
table answers correctly every time. See D11 for the same argument applied to
the decline rules.

GOTCHA: the ordering is `recruiter_email` first, then `from_email` — NOT the
other way round. For a relay message both are non-null and only the extracted
one reaches a human. Preferring the From header would send every relayed reply
into a forwarding address that may already have expired.
"""

from __future__ import annotations

import base64
import logging
import re

from app.gmail.parser import ParsedMessage
from app.llm.schemas import Opportunity

log = logging.getLogger(__name__)

# Domains that route mail through a job board rather than to a person.
# GOTCHA: matched on the registrable domain plus a dot, so `naukri.com` also
# catches `match.naukri.com` but never a company that happens to be called
# `naukrisolutions.io`.
PORTAL_DOMAINS: frozenset[str] = frozenset({
    "naukri.com",
    "linkedin.com",
    "indeed.com",
    "match.indeed.com",
    "glassdoor.com",
    "foundit.in",
    "monster.com",
    "monsterindia.com",
    "shine.com",
    "timesjobs.com",
    "instahyre.com",
    "cutshort.io",
    "hirist.com",
    "iimjobs.com",
})

# Local parts that announce the mailbox discards replies. Substring match,
# because the surrounding decoration varies endlessly: `no-reply`,
# `donotreply`, `jobs-noreply`, `jobalerts-noreply`, `bounce+123`.
NO_REPLY_MARKERS: tuple[str, ...] = (
    "noreply",
    "no-reply",
    "no_reply",
    "donotreply",
    "do-not-reply",
    "do_not_reply",
    "bounce",
    "mailer-daemon",
    "postmaster",
    "notifications",
    "notification",
    "alerts",
    "alert@",
)

_EMAIL_RE = re.compile(r"^[^@\s]+@([^@\s]+)$")


def _domain_of(email: str) -> str | None:
    m = _EMAIL_RE.match(email.strip().lower())
    return m.group(1) if m else None


def is_no_reply_address(email: str | None) -> bool:
    """True if the local part advertises that replies are discarded."""
    if not email:
        return True
    local = email.strip().lower().split("@", 1)[0]
    return any(marker.rstrip("@") in local for marker in NO_REPLY_MARKERS)


def is_portal_address(email: str | None) -> bool:
    """True if this address belongs to a job board rather than an employer."""
    domain = _domain_of(email or "")
    if domain is None:
        return True  # unparseable is not a place we send mail
    return any(
        domain == portal or domain.endswith("." + portal)
        for portal in PORTAL_DOMAINS
    )


def is_replyable(email: str | None) -> bool:
    """True if `email` looks like a mailbox a human reads."""
    if not email or _domain_of(email) is None:
        return False
    return not is_portal_address(email) and not is_no_reply_address(email)


# A plausible domain: labels separated by dots, alphabetic TLD. Used to decide
# whether a base64 decode produced a real domain or coincidental noise.
_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.[a-z]{2,}$")

# Below this length a base64 chunk decodes to something short enough to match
# a domain pattern by accident.
_MIN_B64_CHUNK = 8


def decode_naukri_relay(email: str | None) -> str | None:
    """Recover the recruiter's real address from a Naukri relay local part.

    CONCEPT: Naukri rewrites the From header as `{name}{base64(domain)}@naukri.com`.
      sunita.rao02bm9ydGh3aW5kLmNvbQ==@naukri.com
                  ^^^^^^^^^^^^^^^^^^^^ base64("northwind.com")
      -> sunita.rao02@northwind.com
    The recruiter is reachable directly; the address just has to be decoded.

    WHY this matters: D47 already recovered relayed addresses when the body
    happened to name them, and skipped the message otherwise. Measured over
    200 real messages, that left 13 genuine recruiters unanswered — Northwind,
    Harborlane, Stellarcorp, Crestline, Midwest Consultants and others — every
    one of them reachable, none of them replied to.

    GOTCHA: scan for the LONGEST valid base64 suffix, not the shortest. The
    first version of this walked backwards from the end and produced
    `sunita.rao02bm9y@thwind.com` — a short tail of the encoded domain
    happens to decode to something domain-shaped. That address is deliverable-
    looking and wrong, which is worse than not replying at all.
    """
    if not email or "@" not in email:
        return None
    local, _, domain = email.partition("@")
    if domain.lower() != "naukri.com":
        return None

    for i in range(len(local)):
        chunk = local[i:]
        if len(chunk) < _MIN_B64_CHUNK:
            break
        if not re.fullmatch(r"[A-Za-z0-9+/=]+", chunk):
            continue
        try:
            decoded = base64.b64decode(chunk + "===").decode("ascii").lower()
        except Exception:
            continue
        if _DOMAIN_RE.fullmatch(decoded):
            name = local[:i]
            if name:
                return f"{name}@{decoded}"
    return None


def resolve_reply_target(
    parsed: ParsedMessage,
    opportunity: Opportunity | None,
) -> str | None:
    """Return the address a reply should go to, or None if there isn't one.

    TRACE: at runtime this runs once per message immediately after extraction,
    before embed_jd. Returning None short-circuits the rest of the pipeline —
    no embedding, no dedup lookup, no fit-scoring call, no draft. On a corpus
    where 18 of 35 extracted opportunities were portal alerts, that is roughly
    half the per-message LLM spend avoided on mail that could never be
    answered.
    """
    # B before A — see the module GOTCHA. For a relay message both fields are
    # populated and only this one reaches a person.
    recruiter_email = opportunity.recruiter_email if opportunity else None
    if is_replyable(recruiter_email):
        if not is_replyable(parsed.from_email):
            log.info(
                "reply target recovered from body for message_id=%s: "
                "sender %r is not replyable, using extracted %r",
                parsed.message_id, parsed.from_email, recruiter_email,
            )
        return recruiter_email

    if is_replyable(parsed.from_email):
        return parsed.from_email

    # Last resort before giving up: the sender may be a Naukri relay whose
    # local part encodes the recruiter's real address. Tried AFTER the body
    # lookup because the body is authoritative when present — the decode is a
    # derivation, the body is the recruiter stating their own address.
    decoded = decode_naukri_relay(parsed.from_email)
    if is_replyable(decoded):
        log.info(
            "reply target decoded from naukri relay for message_id=%s: "
            "%r -> %r", parsed.message_id, parsed.from_email, decoded,
        )
        return decoded

    log.info(
        "no reply target for message_id=%s (from=%r, extracted=%r); "
        "skipping before draft",
        parsed.message_id, parsed.from_email, recruiter_email,
    )
    return None


# Subject prefixes mail servers and clients prepend to automatic replies.
# GOTCHA: matched on the SUBJECT, not the body. An out-of-office quotes the
# original message, so body-based detection would fire on the recruiter's own
# words rather than on the auto-reply wrapper around them.
_AUTO_REPLY_MARKERS: tuple[str, ...] = (
    "automatic reply",
    "auto reply",
    "auto-reply",
    "autoreply",
    "out of office",
    "out-of-office",
    "ooo:",
    "away from my desk",
    "undeliverable",
    "delivery status notification",
    "mail delivery failed",
    "returned mail",
    # D79: the phrase Gmail's own web UI shows on a bounce. The Subject header
    # is usually "Delivery Status Notification (Failure)", so this rarely fires
    # here — but it is what a user sees and therefore what some relays put in
    # the actual subject. Cheap to cover, and its absence was a real gap:
    # is_auto_responder("Address not found") returned False.
    "address not found",
)


def is_auto_responder(subject: str | None) -> bool:
    """True if this is a machine-generated reply rather than a human writing.

    WHY this matters more than it looks: an out-of-office is a reply to mail
    WE sent, and its From address is a perfectly real, perfectly replyable
    human mailbox — so every other check here passes it. Answering one puts a
    draft in front of a server that will answer again. The observed case was
    `careers@harborlane.com`, subject "Automatic reply: Experienced AI/ML
    Engineer..." — our own outreach coming back.
    """
    if not subject:
        return False
    lowered = subject.strip().lower()
    return any(marker in lowered for marker in _AUTO_REPLY_MARKERS)
