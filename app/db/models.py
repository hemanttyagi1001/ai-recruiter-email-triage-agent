"""
SQLAlchemy ORM models for the three Phase 0 tables: messages, opportunities,
runs. The design choice worth understanding here is the primary key on
messages — the RFC 5322 Message-ID header, not a surrogate integer, not
Gmail's internal id. That single choice is what makes re-ingesting the same
mailbox idempotent, which is the whole point of Phase 0.

Enum-typed columns store as TEXT with a Python StrEnum for validation, not
as PostgreSQL native ENUM types. Native enums require ALTER TYPE dance to
extend, which is painful in transactional migrations; TEXT + StrEnum lets us
add a new state with a code-only change.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# CONCEPT: pgvector's SQLAlchemy adapter registers a `Vector` column type
# that maps `list[float]` in Python to `vector(N)` in Postgres. Reads come
# back as list[float] (not a numpy array — we don't force that dep).
# GOTCHA: importing this at module top means every process that imports
# app.db.models needs pgvector installed. That's fine — it's in
# pyproject.toml as a hard dep. If we ever want dedup to be an optional
# install, this import moves inside the function that needs it.
from pgvector.sqlalchemy import Vector


def _utcnow() -> datetime:
    # WHY explicit UTC: naive datetimes silently coerce to server-local time
    # somewhere in the driver/DB round-trip. Always store tz-aware.
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# CONCEPT: StrEnum values ARE the on-disk representation. `MessageStatus.FETCHED`
# serialises to the literal string "fetched". This is what lets us store the
# enum as a TEXT column and still get name-based access in Python. Comparisons
# work both ways: `msg.status == MessageStatus.FETCHED` and
# `msg.status == "fetched"` both hold.
class MessageStatus(StrEnum):
    FETCHED = "fetched"
    CLASSIFIED = "classified"
    EXTRACTED = "extracted"
    EXTRACTION_FAILED = "extraction_failed"
    SKIPPED_WRONG_CATEGORY = "skipped_wrong_category"
    # The message was a genuine job pitch, but no human address could be found
    # to answer — a job-board alert digest, or a relay whose body did not name
    # the recruiter. Distinct from SKIPPED_WRONG_CATEGORY, which means "this
    # was not recruitment mail at all": here the content was relevant and we
    # simply have nowhere to send a reply. Keeping them apart is what makes
    # "how much of my inbound is unanswerable portal noise" a query rather
    # than a guess. See D47.
    # WHY no migration: status is a TEXT column with a Python StrEnum (D10)
    # precisely so adding a state stays a code-only change.
    SKIPPED_NO_REPLY_TARGET = "skipped_no_reply_target"
    # We have already sent this recruiter a reply. Terminates before scoring
    # and drafting, so a recruiter who mails five times costs one reply and
    # four cheap classify+extract passes. See D60.
    SKIPPED_ALREADY_REPLIED = "skipped_already_replied"
    # Phase 1 additions.
    DRAFTED = "drafted"
    NEEDS_REVIEW = "needs_review"
    # Phase 2 additions — the interrupt-and-approve lifecycle.
    AWAITING_APPROVAL = "awaiting_approval"      # persisted, paused at interrupt, human review pending
    SENT_TO_GMAIL_DRAFTS = "sent_to_gmail_drafts"  # act ran, Gmail draft created
    REJECTED = "rejected"                          # human rejected via API
    # Phase 5 — the autonomy path (rule-based decline sends without a human).
    # WHY a separate value and not a bool on SENT_TO_GMAIL_DRAFTS: the
    # report/query surface must distinguish intent, not just outcome. A
    # message that ended up in the Drafts folder for a human to send is a
    # fundamentally different event from one the agent sent by itself,
    # and treating them as "sent-with-a-flag" makes every query in
    # cli/report and cli/digest read a bool it might forget to filter on.
    # See D37.
    AUTO_SENT = "auto_sent"


class DraftType(StrEnum):
    DECLINE = "decline"
    INTERESTED = "interested"
    # An off-field role: decline THIS vacancy, state the AI/ML focus, and
    # attach the CV so the recruiter can match future requisitions. Distinct
    # from DECLINE because it is not a refusal — it is a counter-offer, and
    # the report/digest surfaces should be able to count them apart.
    # GOTCHA: only ever set on the fit-score path (rules passed, score below
    # threshold). A rule-fired decline must NEVER become a pivot: two of those
    # rules catch paid-placement and resume-service solicitations, and this
    # shape attaches a CV. See D64.
    # WHY no migration: draft_type is a TEXT column with a Python StrEnum
    # (D10) precisely so adding a value stays a code-only change. "pivot" also
    # fits the String(16) width.
    PIVOT = "pivot"
    # A screening form answered from the profile. Distinct from INTERESTED
    # because it expresses no view on the role — it returns facts that were
    # asked for. Counting them apart is how "how much of my inbound is
    # paperwork" becomes a query. See D67.
    QUESTIONNAIRE = "questionnaire"


class DraftStatus(StrEnum):
    # Phase 1 values — kept for backward compat with pre-Phase-2 rows.
    PENDING_REVIEW = "pending_review"
    QUARANTINED = "quarantined"
    # Phase 2 values. The migration backfills pending_review → awaiting_approval.
    AWAITING_APPROVAL = "awaiting_approval"
    SENT_TO_GMAIL_DRAFTS = "sent_to_gmail_drafts"
    REJECTED = "rejected"
    # Phase 5 — auto path terminal state (mirrors MessageStatus.AUTO_SENT).
    AUTO_SENT = "auto_sent"


class Category(StrEnum):
    NEW_ROLE_PITCH = "new_role_pitch"
    FOLLOWUP_EXISTING_THREAD = "followup_existing_thread"
    INTERVIEW_SCHEDULING = "interview_scheduling"
    DOCUMENT_REQUEST = "document_request"
    PAID_PLACEMENT_SOLICITATION = "paid_placement_solicitation"
    RESUME_SERVICE_SOLICITATION = "resume_service_solicitation"
    NOT_RECRUITMENT = "not_recruitment"


# Categories that proceed to extraction and drafting. Kept as a module-level
# constant so it can be imported by the classify → extract edge in the graph
# without re-listing (and drifting from) the enum.
#
# CONCEPT: why FOLLOWUP_EXISTING_THREAD was removed (D55).
#   It held four different situations under one label — a CTC negotiation, a
#   rejection, a closure, and a technical screening questionnaire — and the
#   drafting layer has two templates. Measured on real mail, 4 of 5 follow-up
#   drafts were wrong: we asked "could you share the full JD?" of a recruiter
#   who had just written "we do not have any openings that match your skill
#   set", and we ignored a screening questionnaire entirely.
#
#   The deeper reason is architectural, and no amount of template tuning fixes
#   it: the agent has NO THREAD HISTORY. It sees exactly one message. A cold
#   inbound pitch is self-contained, so that is enough. A reply inside a
#   conversation the candidate started is only meaningful against context the
#   agent cannot see, so anything it writes reads as a non-sequitur.
#
#   Follow-ups are still classified, extracted and stored — they simply stop
#   before drafting and wait for a human.
# GOTCHA: re-adding this without also giving the pipeline thread history would
# reintroduce the same failure, LLM-drafted or not.
EXTRACTABLE_CATEGORIES: frozenset[Category] = frozenset(
    {Category.NEW_ROLE_PITCH}
)


class WorkModel(StrEnum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"


class EmploymentType(StrEnum):
    PERMANENT = "permanent"
    C2H = "c2h"
    CONTRACT = "contract"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=RunStatus.RUNNING)

    messages_fetched: Mapped[int] = mapped_column(Integer, default=0)
    messages_new: Mapped[int] = mapped_column(Integer, default=0)
    messages_classified: Mapped[int] = mapped_column(Integer, default=0)
    messages_extracted: Mapped[int] = mapped_column(Integer, default=0)
    messages_extraction_failed: Mapped[int] = mapped_column(Integer, default=0)

    classify_tokens_in: Mapped[int] = mapped_column(BigInteger, default=0)
    classify_tokens_out: Mapped[int] = mapped_column(BigInteger, default=0)
    extract_tokens_in: Mapped[int] = mapped_column(BigInteger, default=0)
    extract_tokens_out: Mapped[int] = mapped_column(BigInteger, default=0)
    # Phase 1 additions
    score_tokens_in: Mapped[int] = mapped_column(BigInteger, default=0)
    score_tokens_out: Mapped[int] = mapped_column(BigInteger, default=0)
    rules_declined: Mapped[int] = mapped_column(Integer, default=0)
    messages_scored: Mapped[int] = mapped_column(Integer, default=0)
    messages_needs_review: Mapped[int] = mapped_column(Integer, default=0)
    drafts_created: Mapped[int] = mapped_column(Integer, default=0)
    drafts_quarantined: Mapped[int] = mapped_column(Integer, default=0)
    # Phase 4 additions — embedding token accounting. No _out counterpart
    # because embedding responses carry no completion tokens.
    embed_tokens_in: Mapped[int] = mapped_column(BigInteger, default=0)

    estimated_cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=Decimal("0"))


class Message(Base):
    __tablename__ = "messages"

    # CONCEPT: Idempotency key. RFC 5322 §3.6.4 says every message SHOULD carry
    # a Message-ID header, globally unique, stamped by the originating MTA at
    # composition time. Because we PK on it, a second ingest run trying to
    # INSERT the same message collides on the primary key and we treat that as
    # "already have it, skip." Zero duplicates by construction, not by
    # convention.
    # ALTERNATIVE #1: (subject, from_email) composite key. Rejected: a
    # recruiter sending "Following up on the .NET role" three times produces
    # three legitimately distinct emails that would collapse to one row. Also
    # breaks when ATS platforms rotate sender addresses per campaign.
    # ALTERNATIVE #2: Gmail's internal message id as PK. Rejected: it's
    # provider-specific — swapping to Outlook later would force a data
    # migration. Kept as `gmail_id` (UNIQUE) for a cheap pre-fetch skip during
    # re-ingest, but not the canonical identity.
    # GOTCHA: a tiny fraction of mail arrives without a Message-ID (old MTAs,
    # broken clients). The Gmail parser synthesises one from gmail_id in that
    # case so this column is never null.
    message_id: Mapped[str] = mapped_column(Text, primary_key=True)

    gmail_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)

    from_email: Mapped[str] = mapped_column(String(320))  # 320 = RFC 5321 max
    from_name: Mapped[str | None] = mapped_column(String(320), nullable=True)
    subject: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    body_text: Mapped[str] = mapped_column(Text)
    # WHY JSONB not JSON: JSONB is binary, indexed, faster to query. We do not
    # need to preserve original whitespace / key order of the headers dict.
    raw_headers: Mapped[dict] = mapped_column(JSONB)

    status: Mapped[str] = mapped_column(String(32), default=MessageStatus.FETCHED, index=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    classified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # WHY run_id on message: lets the report CLI answer "what did this run
    # produce" with a single WHERE clause. A message keeps the id of the FIRST
    # run that processed it — subsequent re-ingests skip it on the message_id
    # collision above, so this field is stable.
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id"), index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    opportunity: Mapped[Opportunity | None] = relationship(
        back_populates="message", uselist=False, cascade="all, delete-orphan"
    )


class Opportunity(Base):
    __tablename__ = "opportunities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # WHY UNIQUE on message_id: at most one opportunity per email in Phase 0.
    # Enforcing it at the DB level means a retry-bug that double-inserts is
    # caught by Postgres before it corrupts the report, not by a downstream
    # dedupe query trying to guess the "right" row.
    message_id: Mapped[str] = mapped_column(
        Text, ForeignKey("messages.message_id"), unique=True
    )

    # Every extracted field is Optional. The extractor is instructed to return
    # null for anything not explicitly stated in the email. Optional here means
    # "the email didn't say," not "we couldn't parse."
    company: Mapped[str | None] = mapped_column(Text, nullable=True)
    end_client: Mapped[str | None] = mapped_column(Text, nullable=True)
    role_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    location: Mapped[str | None] = mapped_column(Text, nullable=True)
    work_model: Mapped[str | None] = mapped_column(String(16), nullable=True)
    employment_type: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # WHY LPA (lakhs per annum) as the storage unit: Indian recruiter mail
    # speaks LPA uniformly. Normalising once at extraction time beats parsing
    # "18-22 LPA" vs "1800000" vs "22 lac" at every query.
    ctc_min_lpa: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    ctc_max_lpa: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default="INR")

    notice_period: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recruiter_name: Mapped[str | None] = mapped_column(String(320), nullable=True)
    recruiter_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    recruiter_phone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_platform: Mapped[str | None] = mapped_column(String(64), nullable=True)
    jd_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # WHY nullable + fixed dim: extractions with no jd_text (or below the
    # min_jd_chars floor) produce no embedding — same null-means-unstated
    # discipline the rest of Opportunity follows. Dim is 1536 to match
    # text-embedding-3-small; a schema mismatch on insert raises a
    # Postgres error, not a silent truncation. See D23.
    jd_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536), nullable=True
    )

    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    message: Mapped[Message] = relationship(back_populates="opportunity")


class Draft(Base):
    """A generated reply awaiting human review.

    In Phase 1 nothing sends. Each draft is either pending_review (approved
    by validator, waiting for human sign-off) or quarantined (validator
    rejected it — reason is on quarantine_reason). Whichever the case,
    the raw body_text is preserved untouched — no silent editing.
    """

    __tablename__ = "drafts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # UNIQUE: one draft per message in Phase 1.
    message_id: Mapped[str] = mapped_column(
        Text, ForeignKey("messages.message_id", ondelete="CASCADE"), unique=True
    )
    # Nullable — a decline draft may reach here from a rule fire path even
    # if the opportunity insert failed for some other reason. Keeping the
    # FK optional lets us still record the draft.
    opportunity_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("opportunities.id"), nullable=True
    )

    draft_type: Mapped[str] = mapped_column(String(16))         # decline | interested
    body_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), index=True)  # pending_review | quarantined
    quarantine_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Why-this-verdict provenance. Populated whichever path led here.
    rule_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rule_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    fit_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fit_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    fit_uncertain: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Phase 2 additions — the human approval / act outcome.
    # `approval_reason` = human's note on approve, or (required) reason on reject.
    # `gmail_draft_id` = Gmail's draft id returned from drafts.create.
    # `resolved_at` = when the human decided (approve or reject).
    approval_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    gmail_draft_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Phase 5 — distinguishes autonomous send (rule-based decline) from
    # human-approved send. Column-not-status because we sometimes want
    # to query "everything that went to Gmail today" without caring how
    # it got there; other times we want "only the ones we didn't
    # supervise." Having both a status value AND this flag gives the
    # digest a fast index scan without a full-table filter. See D37.
    auto_actioned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # The address this reply actually went to — the RESOLVED target, which is
    # frequently NOT messages.from_email (D47 prefers the recruiter address
    # found in the body; D59 base64-decodes Naukri relays). Recorded so
    # "have we already replied to this person" is an indexed lookup rather
    # than a re-derivation over history. See D60 and migration 0006.
    # GOTCHA: nullable, because every row written before 0006 has none. A NULL
    # never matches a dedup check, so old rows fail toward sending rather than
    # toward silence — see app/cli/backfill_reply_to.py for the repair.
    reply_to_email: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DuplicateFlag(Base):
    """A single "opportunity X looks like opportunity Y" record.

    One row per (new_opportunity, matched_opportunity) pair whose cosine
    similarity crossed the configured threshold. Multiple rows per new
    opportunity are expected — a role pitched by three vendors produces
    two flags on the third arrival, each pointing at a prior sibling.

    CONCEPT: the flag is metadata for a human, not a routing signal.
    Nothing in the graph reads this table to decide what to do with a
    message; it exists purely to surface "you've seen something like
    this before" in the review UI. See D28 for the cost-asymmetry
    argument behind flag-not-suppress.
    """

    __tablename__ = "duplicate_flags"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    opportunity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("opportunities.id", ondelete="CASCADE"),
        index=True,
    )
    matched_opportunity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("opportunities.id", ondelete="CASCADE"),
    )
    # WHY NUMERIC(6,5) not FLOAT: similarity is bounded [-1, 1] and the
    # calibration script compares thresholds at 0.01 resolution. Fixed-point
    # avoids the "0.85 stored as 0.8499999" surprise when threshold code
    # says `>= 0.85`. Storage is 8 bytes anyway.
    similarity: Mapped[Decimal] = mapped_column(Numeric(6, 5))
    flagged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class DeadLetter(Base):
    """A failure that exhausted the retry policy — infra-level, not domain.

    CONCEPT: what belongs here vs. what belongs on messages/opportunities.
      Domain failures (extractor produced garbage, validator quarantined a
      draft) are business events with their own columns on messages /
      drafts. Infra failures (Azure returned 401, Gmail 429'd us five
      times, DB connection dropped) are events about our RUNTIME, not
      about a specific message's content. Mixing them onto messages
      would force every domain query to filter out infra noise.

    A row here means: retry_external gave up. Somebody (the digest, an
    operator) needs to look at it. Nothing in the pipeline auto-recovers
    from a dead_letter — the message either has a status row saying
    "extraction_failed" (domain) or has no row at all (infra failure
    happened before persist), and the human decides whether to re-ingest.
    """

    __tablename__ = "dead_letters"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # WHY nullable: some failures (list_message_ids 401, get_message 5xx
    # before we could parse) occur before we know the RFC 5322 Message-ID.
    # SET NULL on cascade means deleting a message doesn't cascade-delete
    # its historical dead-letters — those are records of what went wrong,
    # keeping them past the message's lifetime is the point.
    message_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("messages.message_id", ondelete="SET NULL"), nullable=True
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    node: Mapped[str] = mapped_column(String(64))
    error_class: Mapped[str] = mapped_column(String(128))
    error_message: Mapped[str] = mapped_column(Text)
    # attempt count, elapsed_ms, external request_id if any — JSONB for
    # queryability across the corpus of failures.
    error_details: Mapped[dict] = mapped_column(JSONB, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )


class SystemFlag(Base):
    """Single-row-per-key config the pipeline consults at runtime.

    Today one key: `sends_halted`. Reading is done via
    app.kill_switch.is_send_halted() which queries by primary key — one
    row, one column, near-zero latency. Writing is a manual UPDATE via
    app/cli/halt.py or a plain psql session.

    CONCEPT: the check location matters more than the storage.
      Reading this at process start or in a cached form defeats the
      purpose (no live effect on the running process). The kill switch
      is meaningful only because the query fires immediately before
      each Gmail write. See D36.
    """

    __tablename__ = "system_flags"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    updated_by: Mapped[str | None] = mapped_column(String(128), nullable=True)


class AgentActivity(Base):
    """The one table to read when asking "what is the agent doing?".

    CONCEPT: why this table exists when six others already hold state.
      The existing tables are all NOUNS — a message, a draft, a run, a
      failure. Each answers "what is the current state of X". None answers
      "what happened, in order, and when", which is the question you
      actually ask when something looks wrong. Reconstructing a narrative
      meant joining four tables and then reading container logs for the
      parts that were never persisted at all.

    CONCEPT: the data was already being collected and thrown away.
      Every pipeline node already emits a NodeEvent (node, at, duration_ms,
      outcome) into TriageState["events"], merged with an operator.add
      reducer. That trail was serialised into LangGraph's checkpoint blob
      and never surfaced — queryable only by deserialising checkpoints.
      This table is where it gets flushed. See app/activity.py.

    WHY append-only with no status column: a log you can UPDATE is a log you
    can no longer trust as a record of what happened. Corrections go in as
    new rows.

    GOTCHA: `message_id` is deliberately NOT a foreign key, unlike the one on
    dead_letters. Activity rows are written from their own transaction and
    can describe a message before its row is committed (or one that was
    later deleted). An audit log that can fail on a referential race is an
    audit log that goes missing exactly when the system is misbehaving.
    """

    __tablename__ = "agent_activity"

    # BigInteger + autoincrement → BIGSERIAL. WHY not a UUID like the other
    # tables: this table is read in time order by a human far more often than
    # it is joined, and a monotonic id makes "everything after row N" a
    # trivial cursor. Nothing external references these ids.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    # Null for supervisor-level rows (cycle start/finish, scope preflight)
    # which are about the process, not about any one message.
    message_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    # info / warning / error. WHY a string and not the stdlib's int levels:
    # `WHERE level = 'error'` is what you will type at 2am.
    level: Mapped[str] = mapped_column(String(8), default="info")

    # Which part of the system spoke: a graph node (classify, auto_send) or a
    # process-level component (watch, ingest, gmail).
    node: Mapped[str] = mapped_column(String(32), index=True)

    # A short machine-readable code — cycle_started, draft_created,
    # quarantined, dead_lettered, infra_outage_aborted, scope_missing.
    # WHY separate from `outcome`: codes are for filtering and counting,
    # prose is for reading. Mixing them means grepping a text column forever.
    event: Mapped[str] = mapped_column(String(48), index=True)

    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Structured extras — rule name, fit score, error class, gmail ids.
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        # CONCEPT: this constraint is what makes flushing safe to repeat.
        #   The events list accumulates across a message's whole graph run,
        #   so persist_pending and a later persist_final both see the early
        #   events. Rather than track what has been flushed, every write is
        #   ON CONFLICT DO NOTHING against this key — replaying a flush is
        #   free, and no event can be recorded twice.
        # GOTCHA: in Postgres NULLs compare as distinct in a UNIQUE
        #   constraint, so supervisor rows (message_id IS NULL) never collide
        #   with each other. That is correct here — they are written once by
        #   an explicit call, not replayed from an accumulating list.
        UniqueConstraint("message_id", "node", "at", name="uq_agent_activity_event"),
        # The query you will actually run: newest first, usually filtered.
        Index("ix_agent_activity_at_desc", at.desc()),
    )
