"""
Dedup lookup + node tests — Phase 4 step 3.

Three of the four cases exercise `find_similar_opportunities` directly
against a real pgvector-enabled DB: it's the piece with all the
interesting behaviour (vector distance, HNSW usable, lookback join). The
fourth exercises the node's skip branch for a missing embedding — no DB
needed for that path.

WHY test the lookup function separately from the node: the node is
orchestration (state read, event emit, error catch); the lookup is the
"is this vector near any prior vector inside a 60-day window" logic. If
these were one blob of code, every test would set up the graph state
just to prod SQL. Splitting them means each test asserts what it cares
about with the minimum scaffolding.

GOTCHA: these tests skip unless TEST_DATABASE_URL is set in the
environment before pytest runs, and unless that DB has the pgvector
extension (conftest's _engine fixture runs CREATE EXTENSION IF NOT
EXISTS on session start).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.candidate import CandidateProfile
from app.db.models import Message, MessageStatus, Opportunity, Run, RunStatus
from app.dedup.lookup import find_similar_opportunities
from app.dedup.nodes import make_dedup_check_node


# ============================================================================
# Fixtures — build a prior opportunity with a chosen jd_embedding + received_at
# ============================================================================


EMBED_DIM = 1536


def _axis_vector(axis: int, dim: int = EMBED_DIM) -> list[float]:
    """A unit basis vector — 1.0 on `axis`, 0.0 elsewhere.

    Two axis vectors on different axes are perfectly orthogonal (cosine
    similarity = 0). Two axis vectors on the same axis are identical
    (cosine = 1). Using these keeps the arithmetic obvious — no floating
    point noise from arbitrary directions.
    """
    v = [0.0] * dim
    v[axis] = 1.0
    return v


def _skewed_vector(main_axis: int, side_axis: int, side_weight: float,
                   dim: int = EMBED_DIM) -> list[float]:
    """A vector mostly on `main_axis` with a small `side_weight` on `side_axis`.

    Cosine similarity with the pure `main_axis` unit vector is
    1 / sqrt(1 + side_weight**2). Used to build a "very similar but not
    identical" query vector for the ≥ threshold test.
    """
    v = [0.0] * dim
    v[main_axis] = 1.0
    v[side_axis] = side_weight
    return v


@pytest.fixture
def profile() -> CandidateProfile:
    """Minimal candidate.toml stub — same shape used in test_rules."""
    return CandidateProfile.model_validate(
        {
            "candidate": {
                "name": "Test User",
                "total_years": 10,
                "relevant_years": 8,
                "stack": ".NET",
                "current_ctc_lpa": 30,
                "expected_ctc_lpa": 45,
                "notice_period": "30 days",
                "current_location": "Bangalore",
                "preferred_location": "Bangalore or remote",
                "employment_status": "Employed",
            },
            "rules": {"ctc_floor_lpa": 38.0},
            "scoring": {"fit_threshold": 65},
            "drafts": {"max_length_chars": 3000},
            "dedup": {
                "lookback_days": 60,
                "similarity_threshold": 0.85,
                "min_jd_chars": 100,
                "max_candidates_returned": 5,
            },
        }
    )


def _insert_prior(session, *, embedding, received_at, gmail_suffix: str):
    """Insert one message + opportunity pair with a given embedding + timestamp.

    Returns the opportunity's uuid so tests can assert candidate.opportunity_id
    against the specific row they created.
    """
    run = Run(status=RunStatus.SUCCEEDED)
    session.add(run)
    session.flush()

    mid = f"<prior-{gmail_suffix}@x>"
    session.add(
        Message(
            message_id=mid,
            gmail_id=f"g-{gmail_suffix}",
            thread_id=f"thr-{gmail_suffix}",
            from_email="r@x.com",
            from_name="R",
            subject="prior role",
            received_at=received_at,
            body_text="prior body",
            raw_headers={},
            run_id=run.id,
            status=MessageStatus.SENT_TO_GMAIL_DRAFTS,
        )
    )
    opp = Opportunity(
        id=uuid4(),
        message_id=mid,
        role_title="Prior role",
        jd_text="a" * 200,
        jd_embedding=embedding,
    )
    session.add(opp)
    session.flush()
    return opp.id


# ============================================================================
# Direct tests of find_similar_opportunities
# ============================================================================


def test_flags_similar_prior_opportunity(db_session):
    """A prior opp whose vector is nearly-identical to the query vector
    is returned with similarity above the default 0.85 threshold."""
    prior_id = _insert_prior(
        db_session,
        embedding=_axis_vector(0),
        received_at=datetime.now(timezone.utc) - timedelta(days=1),
        gmail_suffix="similar",
    )

    # Query with a vector very close to the prior: main axis 0, tiny
    # weight on axis 1. Cosine similarity = 1/sqrt(1.0001) ≈ 0.99995.
    query = _skewed_vector(main_axis=0, side_axis=1, side_weight=0.01)

    candidates = find_similar_opportunities(
        db_session,
        query,
        threshold=0.85,
        lookback_days=60,
        top_k=5,
    )

    assert len(candidates) == 1
    assert candidates[0].opportunity_id == prior_id
    # The exact value is close to 1; assert loosely above 0.99 rather
    # than pinning the specific arithmetic — pgvector's internal float
    # handling could shift the last few digits without meaning anything.
    assert float(candidates[0].similarity) > 0.99


def test_ignores_dissimilar_prior_opportunity(db_session):
    """A prior opp whose vector is orthogonal to the query vector produces
    no candidates — cosine similarity is 0, well below threshold."""
    _insert_prior(
        db_session,
        embedding=_axis_vector(0),
        received_at=datetime.now(timezone.utc) - timedelta(days=1),
        gmail_suffix="orthogonal",
    )

    # Query on a different axis — cosine similarity with prior = 0.
    query = _axis_vector(5)

    candidates = find_similar_opportunities(
        db_session,
        query,
        threshold=0.85,
        lookback_days=60,
        top_k=5,
    )

    assert candidates == []


def test_lookback_window_excludes_older_opportunities(db_session):
    """The 60-day window must include opps at 59 days ago and exclude
    opps at 61 days ago, even when their vectors match identically.

    This is the specific correctness claim behind the JOIN messages on
    received_at — the lookup is scoped to recent context, not to every
    opportunity ever seen."""
    now = datetime.now(timezone.utc)
    recent_id = _insert_prior(
        db_session,
        embedding=_axis_vector(0),
        received_at=now - timedelta(days=59),
        gmail_suffix="recent",
    )
    _insert_prior(
        db_session,
        embedding=_axis_vector(0),
        received_at=now - timedelta(days=61),
        gmail_suffix="stale",
    )

    query = _axis_vector(0)

    candidates = find_similar_opportunities(
        db_session,
        query,
        threshold=0.85,
        lookback_days=60,
        top_k=5,
    )

    # Only the 59-day-old opp comes back. Both have cosine sim = 1 with
    # the query, so the exclusion is purely on the received_at filter,
    # which is the property this test is guarding.
    returned_ids = [c.opportunity_id for c in candidates]
    assert returned_ids == [recent_id]


# ============================================================================
# Node-level test — the skip branch that doesn't touch the DB
# ============================================================================


def test_node_no_ops_when_jd_embedding_missing(profile, parsed_factory):
    """When embed_jd was skipped (or errored), jd_embedding is absent from
    state. dedup_check must return empty candidates without hitting the
    DB — a DB round-trip on every non-embedded message is waste."""
    node = make_dedup_check_node(profile)

    parsed = parsed_factory()
    # No jd_embedding in state — simulates the embed_jd skip/error branch.
    state = {"parsed": parsed}

    result = node(state)

    assert result["duplicate_candidates"] == []
    # The event outcome should announce WHY we skipped so the audit trail
    # distinguishes this from "ran and found nothing." The two look
    # identical downstream (both produce zero flags) but they mean very
    # different things for the human debugging why a duplicate wasn't
    # flagged.
    assert len(result["events"]) == 1
    assert "no jd_embedding" in result["events"][0].outcome
