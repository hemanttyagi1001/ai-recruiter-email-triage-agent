"""
Who spoke last, and whether the agent should say anything (D81).

The rule under test, in the operator's words: if the last message in the
thread is from the recruiter they are waiting on us, so draft; if the last
message is one we sent, the conversation is answered, so stay quiet.

Why it earned its own module and its own tests: nothing in the database knows
that a human replied from Gmail. `already_replied` (D60) answers from the
drafts table, which records only what the AGENT did. Measured on the live
mailbox, 11 of 12 threads carrying an agent draft already contained a sent
reply — so the guard that was supposed to prevent duplicate outreach was
blind to the most common way outreach actually happens.

`we_spoke_last` takes raw Gmail dicts rather than a client precisely so this
can be exercised exhaustively offline. All tests here are offline.
"""

from __future__ import annotations

import pytest

from app.db.models import MessageStatus
from app.rules.thread_replied import thread_already_answered, we_spoke_last


def msg(date: int, *labels: str) -> dict:
    return {"internalDate": str(date), "labelIds": list(labels)}


INBOUND = ("INBOX", "CATEGORY_PERSONAL")
SENT = ("SENT",)
DRAFT = ("DRAFT",)


# --- The rule ---------------------------------------------------------------


def test_recruiter_spoke_last_so_we_should_draft():
    assert we_spoke_last([msg(100, *INBOUND)]) is False


def test_we_spoke_last_so_we_should_not_draft():
    assert we_spoke_last([msg(100, *INBOUND), msg(200, *SENT)]) is True


def test_a_follow_up_after_our_reply_gets_answered():
    """The case that makes 'last message' the right rule rather than 'any
    reply exists'. HR writes, we answer, HR writes again — they are waiting."""
    thread = [msg(100, *INBOUND), msg(200, *SENT), msg(300, *INBOUND)]
    assert we_spoke_last(thread) is False


def test_a_long_conversation_ending_with_us_stays_quiet():
    thread = [
        msg(100, *INBOUND), msg(200, *SENT),
        msg(300, *INBOUND), msg(400, *SENT),
    ]
    assert we_spoke_last(thread) is True


def test_order_in_the_list_does_not_matter_only_the_date():
    """Gmail does not promise ordering, and sorting by position rather than by
    internalDate would make the answer depend on API response order."""
    scrambled = [msg(300, *INBOUND), msg(100, *INBOUND), msg(200, *SENT)]
    assert we_spoke_last(scrambled) is False


# --- Drafts are not answers -------------------------------------------------


def test_our_own_unsent_draft_does_not_count_as_a_reply():
    """THE trap. An agent draft sits in the thread and is the newest message.
    Counting it would make the agent's pending draft suppress the very reply
    it is waiting to become — the feature would disable itself."""
    thread = [msg(100, *INBOUND), msg(200, *DRAFT)]
    assert we_spoke_last(thread) is False


def test_a_draft_newer_than_our_sent_reply_still_reads_as_answered():
    thread = [msg(100, *INBOUND), msg(200, *SENT), msg(300, *DRAFT)]
    assert we_spoke_last(thread) is True


def test_a_thread_of_nothing_but_drafts_is_unanswered():
    assert we_spoke_last([msg(100, *DRAFT), msg(200, *DRAFT)]) is False


# --- Degenerate input -------------------------------------------------------


@pytest.mark.parametrize("messages", [None, []])
def test_no_messages_means_unanswered_not_answered(messages):
    """Silence is not evidence. Returning True on an empty thread would
    suppress a reply on no information at all."""
    assert we_spoke_last(messages) is False


def test_a_message_with_no_date_cannot_become_the_last_word():
    """A malformed message sorts to the beginning, so it can never decide."""
    thread = [msg(100, *INBOUND), {"labelIds": ["SENT"]}]
    assert we_spoke_last(thread) is False


def test_missing_labels_do_not_crash():
    assert we_spoke_last([{"internalDate": "100"}]) is False


# --- The Gmail wrapper fails open -------------------------------------------


class BoomGmail:
    def thread_messages(self, thread_id):
        raise RuntimeError("gmail is down")


class FakeGmail:
    def __init__(self, messages):
        self._messages = messages
        self.asked = []

    def thread_messages(self, thread_id):
        self.asked.append(thread_id)
        return self._messages


def test_a_gmail_failure_assumes_not_answered():
    """Fails OPEN, deliberately, even though the feature exists to suppress
    drafts. Failing closed would silently drop every message during an outage
    with no reply and no draft — losing opportunities with nothing to show.
    A stray draft is deleted in one click."""
    assert thread_already_answered(BoomGmail(), "t1") is False


def test_no_thread_id_short_circuits_without_calling_gmail():
    gmail = FakeGmail([msg(100, *SENT)])
    assert thread_already_answered(gmail, None) is False
    assert gmail.asked == []


def test_the_wrapper_passes_the_thread_id_through():
    gmail = FakeGmail([msg(100, *INBOUND), msg(200, *SENT)])
    assert thread_already_answered(gmail, "thread-42") is True
    assert gmail.asked == ["thread-42"]


# --- The ingest node ---------------------------------------------------------


class _NoRowSession:
    def get(self, *_a, **_kw):
        return None


def _no_row_session():
    from contextlib import contextmanager

    @contextmanager
    def cm():
        yield _NoRowSession()

    return cm()


def test_ingest_skips_an_answered_thread_before_classify(parsed_factory, monkeypatch):
    """The saving, asserted as behaviour: an answered thread must terminate at
    ingest, so it never reaches the classify or extract LLM calls."""
    import app.pipeline.ingest_node as ingest_node

    monkeypatch.setattr(ingest_node, "session_scope", _no_row_session)
    node = ingest_node.make_ingest_node(
        FakeGmail([msg(100, *INBOUND), msg(200, *SENT)])
    )
    out = node({"parsed": parsed_factory()})
    assert out["final_status"] == MessageStatus.SKIPPED_THREAD_ANSWERED


def test_ingest_proceeds_when_the_recruiter_spoke_last(parsed_factory, monkeypatch):
    import app.pipeline.ingest_node as ingest_node

    monkeypatch.setattr(ingest_node, "session_scope", _no_row_session)
    node = ingest_node.make_ingest_node(FakeGmail([msg(100, *INBOUND)]))
    out = node({"parsed": parsed_factory()})
    assert "final_status" not in out


def test_a_bounce_is_still_caught_before_the_gmail_call(parsed_factory, monkeypatch):
    """Ordering: the free checks run first. A bounce must not cost a Gmail
    round-trip to discover."""
    import app.pipeline.ingest_node as ingest_node

    monkeypatch.setattr(ingest_node, "session_scope", _no_row_session)
    gmail = FakeGmail([msg(100, *INBOUND)])
    node = ingest_node.make_ingest_node(gmail)
    out = node({"parsed": parsed_factory(from_email="mailer-daemon@googlemail.com")})
    assert out["final_status"] == MessageStatus.SKIPPED_UNDELIVERABLE
    assert gmail.asked == []


def test_without_a_gmail_client_the_check_is_skipped(parsed_factory, monkeypatch):
    import app.pipeline.ingest_node as ingest_node

    monkeypatch.setattr(ingest_node, "session_scope", _no_row_session)
    out = ingest_node.make_ingest_node(None)({"parsed": parsed_factory()})
    assert "final_status" not in out
