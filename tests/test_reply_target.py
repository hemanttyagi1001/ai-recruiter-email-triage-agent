"""
Reply-target resolution (D47).

Every address in this file is real — taken from the 118 messages ingested on
2026-08-21. That matters more than usual here: the failure this guards against
is not a crash but a correctly-formatted reply delivered to a mailbox that
throws it away, or worse, sent to an opaque relay while the recruiter's actual
address sat in the body unused.
"""

from __future__ import annotations

import pytest

from app.llm.schemas import Opportunity
from app.rules.reply_target import (
    is_no_reply_address,
    is_portal_address,
    is_replyable,
    resolve_reply_target,
)
from tests.conftest import make_parsed


# --- Category A: a person at a company ---------------------------------------

DIRECT = [
    "ritika.s@brightpathtech.in",
    "hr28@quickapply.com",
    "careers@harborlane.com",
    "contact@mkexports.co.in",
    "dineshhr@gmail.com",
    "asha@midwestconsultants.net",
    "deepa@crestlinestaffing.com",
]


@pytest.mark.parametrize("email", DIRECT)
def test_direct_addresses_are_replyable(email):
    assert is_replyable(email), f"{email} should be replyable"


# --- Category C: nobody is there ---------------------------------------------

UNREPLYABLE = [
    "noreply@glassdoor.com",
    "jobalerts-noreply@linkedin.com",
    "jobs-noreply@linkedin.com",
    "donotreply@match.indeed.com",
    "naukrialerts@naukri.com",
    "opportunities@foundit.in",
    # Portal relay: a real human wrote it, but THIS address is a forwarder.
    "ashabWlkd2VzdGNvbnN1bHRhbnRzLm5ldA==@naukri.com",
]


@pytest.mark.parametrize("email", UNREPLYABLE)
def test_portal_and_noreply_addresses_are_not_replyable(email):
    assert not is_replyable(email), f"{email} must not receive a reply"


# --- The two predicates, separately ------------------------------------------


def test_portal_matching_is_domain_anchored():
    """`naukri.com` must not match a company merely containing that string.

    GOTCHA this pins: a naive substring check would classify
    `hr@naukrisolutions.io` as a portal and silently drop a real recruiter.
    """
    assert is_portal_address("x@naukri.com")
    assert is_portal_address("x@match.indeed.com")       # subdomain
    assert not is_portal_address("hr@naukrisolutions.io")
    assert not is_portal_address("hr@mylinkedin-clone.com")


def test_no_reply_detection_covers_decorated_local_parts():
    assert is_no_reply_address("noreply@x.com")
    assert is_no_reply_address("jobs-noreply@x.com")
    assert is_no_reply_address("do-not-reply@x.com")
    assert not is_no_reply_address("ritika.s@brightpathtech.in")


def test_malformed_addresses_are_never_replyable():
    # An address we cannot parse is not a place we send mail.
    for bad in ["", None, "not-an-email", "two@at@signs.com", "@nolocal.com"]:
        assert not is_replyable(bad), f"{bad!r} must not be replyable"


# --- resolve_reply_target: the ordering that matters -------------------------


def test_relay_sender_resolves_to_extracted_recruiter_email():
    """Category B — the case that motivated this whole module.

    Both fields are populated and they disagree. The From header is a Naukri
    forwarder; the body names the recruiter's real mailbox. Preferring the
    header would mail the forwarder.
    """
    parsed = make_parsed(
        gmail_id="g-relay",
        from_email="ashabWlkd2VzdGNvbnN1bHRhbnRzLm5ldA==@naukri.com",
    )
    opp = Opportunity(
        company="Midwest Consultants",
        role_title=".NET Lead",
        recruiter_email="asha@midwestconsultants.net",
    )
    assert resolve_reply_target(parsed, opp) == "asha@midwestconsultants.net"


def test_extracted_email_wins_even_when_sender_is_also_replyable():
    """Ordering is unconditional, not a fallback.

    When a recruiter mails from a shared `careers@` box but signs off with
    their own address, the personal one is the better reply target.
    """
    parsed = make_parsed(gmail_id="g-both", from_email="careers@harborlane.com")
    opp = Opportunity(
        company="Harborlane", role_title="Engineer",
        recruiter_email="rekha@harborlane.com",
    )
    assert resolve_reply_target(parsed, opp) == "rekha@harborlane.com"


def test_direct_sender_used_when_nothing_extracted():
    parsed = make_parsed(gmail_id="g-direct", from_email="hr28@quickapply.com")
    opp = Opportunity(company="QuickApply", role_title="Engineer")
    assert resolve_reply_target(parsed, opp) == "hr28@quickapply.com"


def test_portal_alert_with_no_extracted_email_returns_none():
    """The 18-of-35 case. None is what stops the pipeline before drafting."""
    parsed = make_parsed(gmail_id="g-alert", from_email="noreply@glassdoor.com")
    opp = Opportunity(company="Newpage", role_title="CDP Engineer")
    assert resolve_reply_target(parsed, opp) is None


def test_unreplyable_extracted_email_falls_back_to_sender():
    """A garbage extraction must not poison a perfectly good sender address."""
    parsed = make_parsed(gmail_id="g-fb", from_email="ritika.s@brightpathtech.in")
    opp = Opportunity(
        company="BrightPath", role_title="Engineer",
        recruiter_email="noreply@brightpathtech.in",
    )
    assert resolve_reply_target(parsed, opp) == "ritika.s@brightpathtech.in"


def test_no_opportunity_still_resolves_from_sender():
    """Extraction can fail while the sender remains perfectly answerable."""
    parsed = make_parsed(gmail_id="g-noopp", from_email="hr28@quickapply.com")
    assert resolve_reply_target(parsed, None) == "hr28@quickapply.com"


# --- auto-responders (D54) ---------------------------------------------------


@pytest.mark.parametrize("subject", [
    "Automatic reply: Experienced AI/ML Engineer (GenAI, LLM, RAG)",
    "Out of Office: Re: JD || AI Architect",
    "AUTO-REPLY: I am away",
    "Undeliverable: your message",
    "Delivery Status Notification (Failure)",
])
def test_auto_responder_subjects_detected(subject):
    """These arrive FROM a real, replyable mailbox — that is the trap.

    An out-of-office is our own outreach bouncing back. Every address check in
    this module passes it, so the subject is the only signal, and answering one
    starts a loop with a mail server.
    """
    from app.rules.reply_target import is_auto_responder
    assert is_auto_responder(subject)


@pytest.mark.parametrize("subject", [
    "Re: Experienced AI/ML Engineer (GenAI, LLM, RAG)",
    "JD || Senior Full Stack AI Engineer || Kochi",
    "Hiring Alert",
    "Confirmation Required – Expected CTC",
])
def test_genuine_subjects_not_flagged_as_auto(subject):
    from app.rules.reply_target import is_auto_responder
    assert not is_auto_responder(subject)


def test_auto_responder_handles_missing_subject():
    from app.rules.reply_target import is_auto_responder
    assert not is_auto_responder(None)
    assert not is_auto_responder("")
