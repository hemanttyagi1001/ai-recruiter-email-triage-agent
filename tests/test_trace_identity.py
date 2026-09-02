"""
Tests for the run name, metadata and tags that make a trace findable (D75).

What these cover and why it is a separate file from test_redaction.py: the
redaction tests exercise the anonymizer, which LangSmith applies to run inputs
and outputs. NOTHING here goes through that anonymizer — LangSmith does not
run it over metadata, tags or run names — so these three builders are the only
thing standing between a subject line and a vendor's servers on that channel.
They are therefore tested as a boundary in their own right, and the assertions
run in the SAFE direction: the default posture is asserted more heavily than
the released one.

All offline. No LangSmith key, no network, no database.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.config import settings
from app.observability import tracing


@pytest.fixture
def parsed():
    """A ParsedMessage-shaped stand-in. Only the five identity attributes the
    tracing helpers read are needed, which is the point of them taking `Any`:
    app/api/routes/pending.py hands them a SimpleNamespace built from a DB row
    rather than a ParsedMessage, and one implementation must serve both."""
    return SimpleNamespace(
        message_id="<mid-1@brightpathtech.in>",
        gmail_id="18f0c0ffee",
        subject="Opening for Senior AI/ML Engineer",
        from_email="careers@brightpathtech.in",
        from_name="Brightpath Careers",
        received_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def traced(monkeypatch):
    """Pretend tracing is armed without building a real LangSmith client.

    GOTCHA: `get_client` is lru_cached, so monkeypatching the SETTING alone
    would be silently ignored if anything had already called it in this
    process. Patching the function is what makes these tests independent of
    test ordering.
    """
    monkeypatch.setattr(tracing, "get_client", lambda: object())
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_trace_identifiers", False)


@pytest.fixture
def identified(traced, monkeypatch):
    monkeypatch.setattr(settings, "langsmith_trace_identifiers", True)


# --- Off by default ---------------------------------------------------------


def test_everything_is_empty_when_tracing_is_off(parsed, monkeypatch):
    """The whole feature collapses to nothing without a client, so the call
    sites can splat these unconditionally — the same argument that made
    trace_callbacks() return a list instead of an Optional."""
    monkeypatch.setattr(tracing, "get_client", lambda: None)
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    assert tracing.trace_run_name(parsed) is None
    assert tracing.trace_metadata(parsed) == {}
    assert tracing.trace_tags() == []


def test_identifiers_alone_cannot_leak_anything(parsed, monkeypatch):
    """LANGSMITH_TRACE_IDENTIFIERS=true with tracing off must stay inert.

    WHY assert this rather than trust it: the two flags are independent
    booleans in .env, and "the loose one is set but the safe one is not" is
    exactly the misconfiguration a reader would expect to be dangerous.
    """
    monkeypatch.setattr(tracing, "get_client", lambda: None)
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    monkeypatch.setattr(settings, "langsmith_trace_identifiers", True)
    assert tracing.trace_run_name(parsed) is None
    assert tracing.trace_metadata(parsed) == {}


# --- Traced, identifiers OFF: correlate but do not identify ------------------


def test_strict_metadata_carries_correlation_keys_only(parsed, traced):
    meta = tracing.trace_metadata(parsed)
    # Present: what joins this run to the Postgres rows and records the mode.
    assert meta["message_id"] == "<mid-1@brightpathtech.in>"
    assert meta["gmail_id"] == "18f0c0ffee"
    assert "auto_send_mode" in meta and "draft_mode" in meta
    # Absent: the identifier release did not happen.
    for key in ("email_subject", "email_from", "email_from_name"):
        assert key not in meta


def test_strict_metadata_holds_no_subject_or_sender_anywhere(parsed, traced):
    """Whole-dict check, not key-by-key: a future field carrying the subject
    under a different name would pass the test above and fail this one."""
    dumped = repr(tracing.trace_metadata(parsed))
    assert "Senior AI/ML Engineer" not in dumped
    assert "Brightpath Careers" not in dumped
    assert "careers@brightpathtech.in" not in dumped


def test_strict_run_name_is_none_rather_than_a_useless_placeholder(parsed, traced):
    """A run named the same 200 times is worse than LangGraph's default, which
    at least names the graph. None means "keep yours"."""
    assert tracing.trace_run_name(parsed) is None


def test_run_id_is_stringified(parsed, traced):
    """UUID is not JSON-native and metadata is serialised — same reason
    _finalise_run stringifies Decimal."""
    rid = UUID("00000000-0000-0000-0000-0000000000ab")
    assert tracing.trace_metadata(parsed, rid)["run_id"] == str(rid)


def test_run_id_is_omitted_when_absent(parsed, traced):
    assert "run_id" not in tracing.trace_metadata(parsed)


# --- Traced, identifiers ON: the release ------------------------------------


def test_run_name_names_the_sender_and_the_subject(parsed, identified):
    assert tracing.trace_run_name(parsed) == (
        "Brightpath Careers — Opening for Senior AI/ML Engineer"
    )


def test_run_name_falls_back_to_the_address_when_there_is_no_display_name(
    parsed, identified
):
    """from_name is Optional on ParsedMessage — a bare `<a@b.com>` From header
    parses to None, and a run titled "None — ..." helps nobody."""
    parsed.from_name = None
    assert tracing.trace_run_name(parsed).startswith("careers@brightpathtech.in — ")


def test_run_name_survives_a_missing_subject(parsed, identified):
    parsed.subject = ""
    assert "(no subject)" in tracing.trace_run_name(parsed)


def test_long_subjects_are_truncated_for_the_list_view(parsed, identified):
    parsed.subject = "Urgent requirement " * 30
    name = tracing.trace_run_name(parsed)
    assert name.endswith("…")
    # Sender + separator + 80 chars of subject, comfortably under a row width.
    assert len(name) < 130


def test_identified_metadata_releases_exactly_the_three(parsed, identified):
    meta = tracing.trace_metadata(parsed)
    assert meta["email_subject"] == "Opening for Senior AI/ML Engineer"
    assert meta["email_from"] == "careers@brightpathtech.in"
    assert meta["email_from_name"] == "Brightpath Careers"
    # And nothing else grew. Body text has no business in metadata, and this
    # is the assertion that would notice someone adding it "just for context".
    assert set(meta) == {
        "message_id", "gmail_id", "auto_send_mode", "draft_mode",
        "gmail_label", "email_subject", "email_from", "email_from_name",
    }


def test_metadata_key_names_match_the_redaction_allowlist(parsed, identified):
    """The two channels must state one policy.

    Metadata bypasses the anonymizer, so this pairing is not enforced at
    runtime by anything — it is a convention, and a convention that nothing
    checks is a convention that drifts. This test is the check.
    """
    from app.observability.redaction import IDENTIFIER_KEYS

    released = set(tracing.trace_metadata(parsed)) - set(
        tracing.trace_metadata(_stripped(parsed))
    )
    assert released <= IDENTIFIER_KEYS


def _stripped(parsed):
    """The same message with the identifier fields blanked, used to compute
    which metadata keys exist *because of* the release."""
    return SimpleNamespace(
        message_id=parsed.message_id, gmail_id=parsed.gmail_id,
        subject=None, from_email=None, from_name=None,
    )


# --- Tags -------------------------------------------------------------------


def test_tags_are_low_cardinality_and_carry_no_message_data(parsed, identified):
    """A tag is a coarse click-filter. Anything per-message belongs in
    metadata, or the tag list becomes unusable after a hundred runs."""
    tags = tracing.trace_tags()
    assert all(":" in t for t in tags)
    joined = " ".join(tags)
    for personal in ("Brightpath", "careers@", "Senior AI/ML", "<mid-1"):
        assert personal not in joined
