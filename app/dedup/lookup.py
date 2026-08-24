"""
pgvector nearest-neighbour lookup for the dedup pass.

This module is a single function — `find_similar_opportunities` — plus the
small `DuplicateCandidate` value it returns. Split from `nodes.py` so the
graph node stays orchestration-only (state read, event emit) and this
module owns the SQL. Also lets tests exercise the query without spinning
the whole graph.

CONCEPT: cosine similarity in pgvector, the operator you actually use.
  pgvector exposes three distance operators on `vector` columns:
    <->   L2 (Euclidean) distance
    <#>   negative inner product
    <=>   cosine distance = 1 - cos_sim
  Cosine is the right choice for text embeddings because embedding
  models are trained with a cosine-similarity objective — the DIRECTION
  of the vector is what carries meaning; magnitude drifts with token
  count. Similarity = 1 - <=>. Range: [0, 2] for arbitrary vectors;
  [0, 1] in practice for text embeddings (which are ~normalised).

CONCEPT: why we ORDER BY the distance operator, not the similarity.
  HNSW indexes are built on the distance operator, not on arbitrary
  expressions. `ORDER BY jd_embedding <=> :vec ASC LIMIT :k` uses the
  index; `WHERE 1 - (jd_embedding <=> :vec) >= :threshold` does NOT
  (Postgres has no way to know which rows to inspect). The correct
  pattern is:
      SELECT ..., 1 - (jd_embedding <=> :vec) AS similarity
      ORDER BY jd_embedding <=> :vec ASC
      LIMIT :k
  and then filter by similarity in Python. That's what this function does.

CONCEPT: why we JOIN messages instead of adding received_at to opportunities.
  The received_at we care about is the recruiter email's send time — that
  lives on `messages`, not `opportunities`. Denormalising it onto
  opportunities would let us skip the JOIN but at the cost of a field
  that means "the received_at of the message that produced me" — the
  kind of implicit foreign-key-in-a-column that goes stale silently the
  first time someone touches the message row. The JOIN is cheap
  (opportunities has a unique constraint on message_id — one row per).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class DuplicateCandidate:
    """A single "this new opp looks like an older opp" record.

    similarity is Decimal (not float) to match the storage type on
    duplicate_flags — keeps the round-trip lossless. Values are always
    in [0, 1] for text embedding cosine similarity.
    """

    opportunity_id: UUID
    similarity: Decimal


def find_similar_opportunities(
    session: Session,
    vector: list[float],
    *,
    threshold: float,
    lookback_days: int,
    top_k: int,
) -> list[DuplicateCandidate]:
    """Return the top-K opportunities within `lookback_days` whose jd_embedding
    is at least `threshold` cosine-similar to `vector`.

    Args:
        session: active SQLAlchemy session bound to a pgvector-enabled DB.
        vector: the new JD's embedding (must match the column dim; a
                mismatch raises a pgvector error, not silent truncation).
        threshold: minimum cosine similarity to consider a match. Values
                   in [0, 1]; 0.85 is the pre-calibration default.
        lookback_days: only messages received within this window count.
                       60 is the pre-calibration default; see D27.
        top_k: at most this many neighbours are returned. See D30.

    Empty list means "nothing above threshold" — the caller distinguishes
    "no matches" from "dedup skipped" via the state field being absent
    (skipped) vs. present-but-empty (ran, no matches).
    """
    # WHY compute the cutoff in Python and pass it as a bound param: the
    # alternative is `WHERE m.received_at >= NOW() - interval :days` where
    # `:days` is a string like "60 days". That mixes user-supplied numeric
    # into interval-string parsing on the DB side, which is fragile and
    # awkward to bind. Passing a timestamp is a plain typed parameter.
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # GOTCHA: pgvector requires the parameter cast to `vector` — a raw
    # Python list arrives as an array-of-numeric, and the `<=>` operator
    # doesn't resolve on array types. Formatting the vector as the string
    # `[0.1,0.2,...]` and casting explicitly is the pgvector-recommended
    # pattern for parameterised queries.
    vec_str = "[" + ",".join(f"{f:.8f}" for f in vector) + "]"

    stmt = text(
        """
        SELECT o.id AS opportunity_id,
               1 - (o.jd_embedding <=> CAST(:vec AS vector)) AS similarity
        FROM opportunities o
        JOIN messages m ON m.message_id = o.message_id
        WHERE o.jd_embedding IS NOT NULL
          AND m.received_at >= :cutoff
        ORDER BY o.jd_embedding <=> CAST(:vec AS vector) ASC
        LIMIT :top_k
        """
    )

    rows = session.execute(
        stmt,
        {"vec": vec_str, "cutoff": cutoff, "top_k": top_k},
    ).all()

    # WHY Python-side threshold filter instead of a SQL WHERE: pushing the
    # cutoff into SQL would prevent the HNSW index from being usable for
    # the ORDER BY / LIMIT (see module docstring). The ORDER-then-LIMIT
    # gives us at most `top_k` rows; filtering that tiny list in Python
    # is free compared to the alternative of a seq-scan.
    return [
        DuplicateCandidate(
            opportunity_id=row.opportunity_id,
            similarity=Decimal(str(row.similarity)),
        )
        for row in rows
        if float(row.similarity) >= threshold
    ]
