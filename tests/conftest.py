"""
Test harness.

Two things are load-bearing about this file:

1. The DATABASE_URL override happens BEFORE `app.config` is imported. Since
   `settings` is instantiated at import time, we must swap the env var
   before the first `from app...` line elsewhere. Pytest imports conftest
   before collecting tests, so this runs first.

2. The DB fixture wraps each test in a transaction and rolls back at the
   end. That's how we get test isolation without dropping/creating tables
   per-test — the schema stays put, but each test's writes vanish.
"""

from __future__ import annotations

import os

# --- Env override: MUST happen before any `from app...` import ---
_TEST_DB_URL = os.environ.get("TEST_DATABASE_URL")
if _TEST_DB_URL:
    os.environ["DATABASE_URL"] = _TEST_DB_URL

# Provide harmless defaults for tests that never actually call Azure. Real
# credentials aren't needed because tests use FakeLLM. Setting them keeps
# pydantic-settings validation from failing at import time.
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.invalid")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "test-deployment")

# WHY tests default to autonomy OFF rather than inheriting the production
# default: `off` is the only mode under which the graph's routing is a pure
# function of message content. Under dry_run or on, every clean draft diverts
# to auto_send, which would silently rewrite the expectations of every test
# written against the human-approval path — they would still pass or fail, but
# about a different pipeline than the one they describe. Tests that care about
# autonomy set the mode explicitly (see test_autonomy_routing.py).
os.environ.setdefault("AUTO_SEND_MODE", "off")

# WHY tests default to template drafting: the LLM drafter adds a FOURTH
# structured_completion call per message, and FakeLLM hands out queued
# responses in order. Tests that queue classify/extract/score get their queue
# consumed one step early, and every later assertion fails somewhere far from
# the cause — the observed symptom was "'ClassificationResult' object has no
# attribute 'body_text'" three messages downstream. Tests that exercise LLM
# drafting inject their own client (see test_llm_draft.py).
os.environ.setdefault("DRAFT_MODE", "template")

# --- Safe to import app now ---
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.models import Base, SystemFlag  # noqa: E402
from app.gmail.parser import ParsedMessage  # noqa: E402
from app.llm.client import Usage  # noqa: E402


# --- DB fixtures ---


@pytest.fixture(scope="session")
def _engine():
    if not settings.test_database_url:
        pytest.skip(
            "TEST_DATABASE_URL not configured; skipping DB tests. "
            "Point it at a scratch Postgres to enable."
        )
    engine = create_engine(settings.database_url, future=True)
    # WHY the extension has to be created BEFORE create_all: the
    # Opportunity model declares a `Vector(1536)` column (Phase 4). Its
    # CREATE TABLE emits `... jd_embedding VECTOR(1536) ...`, which the
    # DB rejects unless the pgvector extension is registered. In the
    # production DB Alembic migration 0004 does this; in the test DB we
    # bypass Alembic and rely on Base.metadata, so we replicate the one
    # statement here. `IF NOT EXISTS` makes it idempotent across sessions.
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # Start clean — safer than assuming previous runs cleaned up.
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(_engine):
    """One transaction per test; rolled back at end for isolation.

    GOTCHA: this fixture is only valid for tests whose code-under-test
    shares THIS session. It hands out a session bound to a connection with
    an already-open transaction, so nothing it writes is visible to any
    other connection until that transaction commits — and it never does.
    Application code calling `session_scope()` opens its own connection and
    will not see these rows. Tests that exercise such code either monkeypatch
    `session_scope` to return this session (see test_dead_letter,
    test_idempotency) or must use `committed_db` instead. Calling
    `.commit()` on this session does NOT rescue you: it commits the
    externally-begun transaction, deassociates it, and makes teardown's
    rollback a no-op that emits SAWarning.
    """
    connection = _engine.connect()
    trans = connection.begin()
    session = sessionmaker(bind=connection, autoflush=False, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()


@pytest.fixture
def committed_db(_engine):
    """Session that COMMITS for real; tables truncated afterwards.

    CONCEPT: why a second DB fixture has to exist.
      Test isolation normally comes from wrapping a test in a transaction and
      rolling it back — fast, and it never touches the schema. That trick
      relies on one assumption: everything participating shares a single
      connection. It breaks the moment the code under test opens its own,
      because an uncommitted row is invisible outside the transaction that
      wrote it.

      That is exactly the situation for anything driving the LangGraph
      pipeline. `session_scope()` deliberately opens a fresh session per
      unit of work (see app/db/engine.py), which is correct for production
      and fatal for rollback-isolation. A `Run` seeded by the test is
      invisible to the graph's persist nodes, so their INSERT trips the
      messages.run_id foreign key.

      So: commit for real, and buy isolation by truncating instead. Slower
      per test — a TRUNCATE round-trip rather than an in-memory rollback —
      which is why this is opt-in and `db_session` remains the default.

    WHY TRUNCATE ... CASCADE rather than DELETE or drop/create: CASCADE lets
    us ignore FK ordering entirely, RESTART IDENTITY resets sequences so
    tests can't depend on ids from a previous run, and truncation leaves the
    schema (and the pgvector extension) intact so we don't re-pay create_all.
    """
    session = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        # GOTCHA: truncate on the way OUT, not on the way in. A test that
        # fails mid-way still leaves the DB clean for the next one, and the
        # failing test's rows survive long enough to be inspected if you
        # drop a breakpoint before teardown.
        table_list = ", ".join(t.name for t in Base.metadata.sorted_tables)
        with _engine.begin() as conn:
            conn.execute(text(f"TRUNCATE {table_list} RESTART IDENTITY CASCADE"))


@pytest.fixture
def sends_enabled(committed_db):
    """Seed the kill switch to OFF so outbound paths can actually execute.

    WHY a test must ask for this explicitly: `is_send_halted()` is fail-safe
    — a MISSING system_flags row counts as HALTED, not as permitted (D36).
    That is the correct production default, because a half-migrated database
    must never be read as "cleared to send mail". The consequence for tests
    is that the empty-table starting state means HALTED, so any test that
    asserts a send actually happened has to seed this row first.

    GOTCHA: the failure mode without this is a passing test that proves
    nothing. `test_rejected_thread_does_not_call_gmail` asserted Gmail was
    never called and passed — but it passed because sends were halted, not
    because the rejection path worked. Requesting this fixture is what makes
    such a test prove its actual claim.
    """
    committed_db.merge(SystemFlag(key="sends_halted", value=False))
    committed_db.commit()
    return committed_db


# --- LLM fakes ---


class FakeLLM:
    """Duck-compatible with LLMClient's structured_completion + embed.

    Queue chat responses with .queue(); each call to structured_completion
    pops one. Queue embed responses with .queue_embed(); each call to
    embed pops one.

    WHY separate queues: embed and structured_completion return different
    shapes ((list[float], Usage) vs (BaseModel, Usage)) and are consumed
    in different orders by the graph. One queue would force tests to
    interleave responses precisely, which is brittle. Separate queues
    let tests declare "I don't care about embed" by leaving that queue
    empty and getting a default zero-vector back.
    """

    def __init__(self) -> None:
        self.responses: list[Any] = []
        self.embed_responses: list[Any] = []
        self.calls: list[tuple[type, list[dict]]] = []
        self.embed_calls: list[str] = []

    def queue(self, response_or_exc: Any) -> None:
        self.responses.append(response_or_exc)

    def queue_embed(self, response_or_exc: Any) -> None:
        self.embed_responses.append(response_or_exc)

    def structured_completion(
        self, schema, messages, temperature: float = 0.0
    ):
        self.calls.append((schema, messages))
        if not self.responses:
            raise RuntimeError(
                f"FakeLLM out of queued responses (call #{len(self.calls)}). "
                f"Queue a (parsed, Usage) tuple or an exception."
            )
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def embed(self, text: str):
        self.embed_calls.append(text)
        if not self.embed_responses:
            # WHY default rather than raise: tests that don't care about
            # embedding (e.g., restart tests measuring pause/resume)
            # shouldn't have to queue an embed response per message. A
            # zero-vector-with-zero-usage default is inert — it flows
            # into state and persists as [0]*1536, but no test asserts
            # on it. Tests that DO care about embedding must queue.
            return ([0.0] * 1536, Usage(prompt_tokens=0, completion_tokens=0, model="fake-embed"))
        r = self.embed_responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


# --- ParsedMessage factory ---


def make_parsed(
    *,
    gmail_id: str = "g-1",
    message_id: str | None = None,
    subject: str = "A recruiter reaches out",
    body: str = "We have a great .NET role at Acme in Bangalore, 25-35 LPA.",
    from_email: str = "recruiter@example.com",
    from_name: str | None = "Recruiter One",
    received_at: datetime | None = None,
) -> ParsedMessage:
    return ParsedMessage(
        message_id=message_id or f"<mid-{gmail_id}@example.com>",
        gmail_id=gmail_id,
        thread_id=f"thr-{gmail_id}",
        from_email=from_email,
        from_name=from_name,
        subject=subject,
        received_at=received_at or datetime.now(timezone.utc),
        body_text=body,
        raw_headers={"Message-ID": message_id or f"<mid-{gmail_id}@example.com>"},
    )


@pytest.fixture
def parsed_factory():
    return make_parsed


# --- Usage factory ---


def usage(prompt: int = 100, completion: int = 20, model: str = "gpt-4o-mini") -> Usage:
    return Usage(prompt_tokens=prompt, completion_tokens=completion, model=model)


@pytest.fixture
def usage_factory():
    return usage
