"""phase 4: pgvector extension, jd_embedding column, HNSW index, duplicate_flags

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-19

CONCEPT: what this migration turns on.
  The pgvector image has been running since Phase 0 (see D7), but the
  extension itself was never CREATE'd. This migration flips the switch,
  adds a `vector(1536)` column to opportunities for the JD embedding,
  builds an HNSW index for approximate nearest-neighbour lookup, and
  creates a `duplicate_flags` table that records "opportunity X looks
  like opportunity Y with cosine similarity S." Nothing gates on that
  table — dedup is a hint to the human, never an auto-close (D28).

ALTERNATIVE (IVFFlat vs HNSW): which index type to use.
  Two families of approximate nearest neighbour indexes ship with
  pgvector; we pick HNSW here. The full argument:

  IVFFlat — partitions the vector space with k-means into `lists`
  centroids. A query picks the `probes` nearest centroids and scans
  their members. Build is fast (O(N) after the k-means fit) and memory
  light. But: k-means needs representative training data, roughly on
  the order of `lists` rows, to produce a good partition. Cold-start on
  a small table produces skewed clusters and awful recall. Re-training
  is manual as the data distribution drifts. Query-time knob is
  `probes`: higher = higher recall, higher latency.

  HNSW — builds a multi-layer navigable small-world graph. A query
  descends greedily from the top layer to the bottom. Build is slower
  (O(N log N) with tuning constants `m` = edges per node and
  `ef_construction` = build-time candidate list size). Memory is
  heavier — roughly `m * N * dim * bytes`. But: incremental inserts
  are natural (no retraining), and recall/latency at a given ef_search
  tend to dominate IVFFlat on small-to-medium N. Query-time knob is
  `ef_search`: higher = higher recall, higher latency.

  Chose HNSW because: (a) our workload is single-row inserts as emails
  arrive; there's no batch to k-means-train IVFFlat against without a
  bootstrap phase; (b) at our N (hundreds now, thousands at 100x) HNSW's
  memory cost is trivial; (c) HNSW gives us better recall at the same
  latency point in the region where we actually operate.

  See D26 in DECISIONS.md for the durable version of this argument.

CONCEPT: approximate NN is acceptable for our use case.
  We're producing a flag for a human reviewer, not answering a
  correctness-critical retrieval. Missing a 0.87 match because the graph
  traversal returned 0.85 first is fine — the review queue asymmetry
  (D28) says false-negative is the cheaper failure anyway. If we ever
  need exact results for an audit, we can force a seq scan at query
  time with SET LOCAL enable_indexscan = OFF; the vector column stays
  scannable no matter what the index looks like.

CONCEPT: partial index (WHERE jd_embedding IS NOT NULL).
  Most opportunities *will* have a JD embedding, but some (short or
  missing jd_text) won't. A partial index skips NULL rows entirely,
  which is both faster to build and correct — the null rows can never
  match a similarity query anyway. WHERE clause on the CREATE INDEX
  gets used automatically by the planner when the query includes the
  same predicate (which our dedup query does).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- 1. Enable the extension. Idempotent; safe if it was already on. ---
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- 2. Embedding column on opportunities ---
    # WHY nullable: extractions with no jd_text (or with jd_text below the
    # min_jd_chars floor) produce no embedding. Same principle as the
    # extractor's "null = source did not state it" — we don't manufacture
    # a vector from missing data.
    # GOTCHA: pgvector's `vector(N)` is dimension-checked at write time.
    # Inserting a wrong-length vector raises a Postgres error, not a
    # silent truncation. Good — that's exactly the kind of schema
    # divergence we want loud.
    op.execute("ALTER TABLE opportunities ADD COLUMN jd_embedding vector(1536)")

    # --- 3. HNSW index, partial on non-null vectors ---
    # m=16, ef_construction=64 are pgvector's documented sensible defaults
    # for a broad range of workloads. We are not in a regime where tuning
    # these pays off — the benchmark script (bench/dedup/benchmark.py)
    # measures latency at real and simulated scale, and D30 records whether
    # the index is even earning its cost.
    op.execute(
        "CREATE INDEX ix_opportunities_jd_embedding_hnsw "
        "ON opportunities USING hnsw (jd_embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64) "
        "WHERE jd_embedding IS NOT NULL"
    )

    # --- 4. duplicate_flags table ---
    # WHY a separate table (not a column on opportunities): one new
    # opportunity can match multiple prior opportunities above threshold.
    # Storing the top-K matches keeps the review UI honest — the human
    # sees every above-threshold neighbour, not just the strongest.
    # WHY ON DELETE CASCADE both sides: if either opportunity is deleted
    # (e.g., a manual cleanup), the flag becomes meaningless. Cascading
    # prevents orphan rows.
    # WHY the CHECK (opportunity_id <> matched_opportunity_id): self-match
    # is nonsense; the dedup query already excludes it, but defence in
    # depth at the schema level is cheap.
    op.create_table(
        "duplicate_flags",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "opportunity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("opportunities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "matched_opportunity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("opportunities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("similarity", sa.Numeric(6, 5), nullable=False),
        sa.Column(
            "flagged_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "opportunity_id <> matched_opportunity_id",
            name="ck_duplicate_flags_no_self_match",
        ),
    )
    op.create_index(
        "ix_duplicate_flags_opportunity_id",
        "duplicate_flags",
        ["opportunity_id"],
    )

    # --- 5. Runs: embedding token accounting ---
    # WHY only tokens_in and not tokens_out: embeddings have no output
    # tokens — the response is a fixed-size vector, not a token stream.
    # Keeping the column shape asymmetric (in only) is honest about that
    # rather than pretending completion_tokens exists.
    op.add_column(
        "runs",
        sa.Column(
            "embed_tokens_in",
            sa.BigInteger,
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("runs", "embed_tokens_in")
    op.drop_index("ix_duplicate_flags_opportunity_id", table_name="duplicate_flags")
    op.drop_table("duplicate_flags")
    op.execute("DROP INDEX IF EXISTS ix_opportunities_jd_embedding_hnsw")
    op.execute("ALTER TABLE opportunities DROP COLUMN jd_embedding")
    # WHY we do NOT drop the extension: other things might be using vector
    # by now, and DROP EXTENSION is a heavyweight operation. If you really
    # want it gone, do it manually.
