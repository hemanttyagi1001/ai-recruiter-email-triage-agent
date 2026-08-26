"""
Test harness.

Three things are load-bearing about this file:

1. The DATABASE_URL override happens BEFORE `app.config` is imported. Since
   `settings` is instantiated at import time, we must swap the env var
   before the first `from app...` line elsewhere. Pytest imports conftest
   before collecting tests, so this runs first.

2. The DB fixture wraps each test in a transaction and rolls back at the
   end. That's how we get test isolation without dropping/creating tables
   per-test — the schema stays put, but each test's writes vanish.

3. Every path to a database URL in this file is checked against the
   PRODUCTION url before anything connects. See the incident note below —
   this file holds the only `drop_all` in the repo, and it once ran against
   the wrong database.

GOTCHA (2026-08-24, D62): the guard and the connection used to read
  different values. The skip guard consulted `settings.test_database_url`,
  which pydantic-settings populates from `.env`; the engine then connected to
  `settings.database_url`, which `.env` populates with PRODUCTION. The
  override above was the only thing that reconciled them, and it fired solely
  when TEST_DATABASE_URL was exported into the SHELL — `.env` alone did not
  satisfy it. So `pytest` in a clean shell passed the guard, connected to the
  production database, and dropped every application table. 73 watch cycles
  then failed on `relation "runs" does not exist` before anyone noticed,
  because a poll loop that dies on its first DB write is silent, not loud.
  The repair is structural, not a warning in a docstring: `.env` now satisfies
  the override too, and `_assert_not_production` gates every connection.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _from_dotenv(key: str) -> str | None:
    """Read one key out of `.env` without importing app.config.

    WHY not just use pydantic-settings, which already parses this file: the
    entire purpose of the block below is to rewrite DATABASE_URL *before*
    app.config is imported. Importing app.config to discover what to rewrite
    would instantiate `settings` against the un-rewritten value, which is the
    ordering we are trying to preserve. Twenty lines of parser is the price of
    not having a bootstrap cycle here.
    """
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _db_name(url: str) -> str:
    """The database name from a SQLAlchemy URL, ignoring any `?...` suffix.

    WHY compare on the database name rather than the whole URL string: the
    same database is reachable as `localhost:5432/triage` from the host and
    `host.docker.internal:5432/triage` from a container (see
    docker-compose.yml). Comparing full URLs would call those two different
    databases and wave the destructive path straight through.
    """
    return url.rsplit("/", 1)[-1].split("?", 1)[0]


# --- Env override: MUST happen before any `from app...` import ---
# WHY `.env` is consulted as a fallback for BOTH values: the old version read
# TEST_DATABASE_URL from the process environment only, while the skip guard
# read it via settings (i.e. from `.env`). One source disagreeing with the
# other is exactly what pointed drop_all at production.
_PROD_DB_URL = os.environ.get("DATABASE_URL") or _from_dotenv("DATABASE_URL")
_TEST_DB_URL = os.environ.get("TEST_DATABASE_URL") or _from_dotenv("TEST_DATABASE_URL")


def _assert_not_production(url: str, what: str) -> None:
    """Refuse to hand out a URL that names the production database.

    TRACE: called twice — once at import time on the override below, once in
    the `_engine` fixture immediately before create_engine. The second call is
    not redundant: it re-checks the value that SQLAlchemy will actually
    receive, after settings has resolved it, so a future edit that changes
    where the engine gets its URL still cannot slip past.

    WHY RuntimeError and not pytest.skip: a misconfigured test database is a
    broken environment, and skipping would hide it behind a green run. This
    aborts collection with the reason on screen.
    """
    if _PROD_DB_URL and _db_name(url) == _db_name(_PROD_DB_URL):
        raise RuntimeError(
            f"REFUSING TO RUN: {what} resolves to database "
            f"{_db_name(url)!r}, which is the same database DATABASE_URL "
            f"names. This fixture calls Base.metadata.drop_all(). Point "
            f"TEST_DATABASE_URL at a scratch database (e.g. triage_test) "
            f"before running pytest."
        )


if _TEST_DB_URL:
    _assert_not_production(_TEST_DB_URL, "TEST_DATABASE_URL")
    # WHY the override still exists: application code under test calls
    # `session_scope()`, which reads settings.database_url. Without this the
    # code being tested would write to production even though the fixture's
    # own engine points elsewhere.
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
    # GOTCHA: the guard reads the module-level `_TEST_DB_URL`, NOT
    # `settings.test_database_url`. They look interchangeable and are not —
    # settings resolves through `.env` on its own, so a guard written against
    # it can pass while the override at the top of this file never fired. That
    # divergence is what dropped the production schema on 2026-08-24.
    if not _TEST_DB_URL:
        pytest.skip(
            "TEST_DATABASE_URL not configured; skipping DB tests. "
            "Point it at a scratch Postgres to enable."
        )
    # Connect to the TEST url explicitly rather than to settings.database_url.
    # The override above should have made them identical; asserting that here
    # rather than trusting it means a broken override fails loudly instead of
    # quietly aiming drop_all somewhere else.
    _assert_not_production(_TEST_DB_URL, "TEST_DATABASE_URL")
    _assert_not_production(settings.database_url, "settings.database_url")
    engine = create_engine(_TEST_DB_URL, future=True)
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
