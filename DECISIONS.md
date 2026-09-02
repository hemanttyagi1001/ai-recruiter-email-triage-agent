# Architectural Decisions

One entry per decision. Short: what we chose, what we considered, why, and
what would make us revisit. Newest at the bottom.

## D1 — LangGraph for the Phase 0 pipeline (2026-08-19)
**Decision:** Model classify → extract → persist as a LangGraph `StateGraph`
even though a plain function chain would work at this size.
**Alternatives:** Plain Python function pipeline; explicit orchestrator class.
**Reason:** This is a LangGraph learning project. The primitives (nodes,
edges, shared state, conditional edges) show up honestly only when the
pipeline has a real branch — Phase 0's "extract only for two of seven
categories" gives us that. Growing into Phase 1 (checkpointing, human-in-
the-loop breakpoints) is edge additions, not a rewrite.
**Revisit if:** The graph stays trivially linear through Phase 1 as well.

## D2 — Message-ID as messages primary key (2026-08-19)
**Decision:** `messages.message_id` (RFC 5322 header) is the PK. Gmail's
internal id is stored as `gmail_id` with a UNIQUE index for fast pre-fetch
skips on re-ingest.
**Alternatives:** gmail_id-only PK; (subject, from_email) composite; surrogate UUID.
**Reason:** Message-ID is globally unique per RFC and stable across
forwards, resends, and folder moves. (subject, from) collides on identical
follow-up mails and on ATS-rewritten sender addresses. A gmail-only PK
creates a provider migration hazard.
**Revisit if:** We ingest from a source that does not stamp Message-IDs.
Fallback would be a synthesised hash.

## D3 — Azure OpenAI structured outputs with strict schema (2026-08-19)
**Decision:** `response_format={"type":"json_schema","strict":true}` derived
from `Opportunity.model_json_schema()`.
**Alternatives:** JSON mode + regex fixups; function-calling; free text +
Pydantic parse-with-repair.
**Reason:** Strict schema mode is grammar-constrained decoding server-side —
the model literally cannot emit malformed JSON or wrong field types. Frees
Pydantic to enforce *semantic* rules (nulls, ranges) rather than shape rules.
**Revisit if:** Azure removes strict mode or we need a schema feature it
doesn't support (e.g. discriminated unions).

## D4 — Prefer text/plain, fall back to HTML→text (2026-08-19)
**Decision:** MIME walk takes text/plain if present. Otherwise strip
text/html via BeautifulSoup.
**Alternatives:** Always convert HTML→text; text/plain only.
**Reason:** Text-only skips HTML-only senders (many recruiter templates).
HTML-always throws away recruiter-authored plain-text formatting when both
parts are present.
**Revisit if:** We need structure only HTML preserves (tables of role details).

## D5 — Aggregated cost tracking only in Phase 0 (2026-08-19)
**Decision:** Token/cost totals aggregate into the `runs` row. No per-call
`llm_calls` table.
**Alternatives:** Per-call table with prompt hash, latency, retry flag.
**Reason:** Phase 0 spec asks for aggregates. Per-call granularity is useful
for eval work but belongs in the evaluation phase.
**Revisit if:** We hit a cost surprise we can't attribute, or evals need
per-call latency distributions.

## D6 — Bodies stored in Postgres TEXT (2026-08-19)
**Decision:** `messages.body_text` is a TEXT column; no on-disk blobs.
**Alternatives:** Bodies on disk, path in DB.
**Reason:** One fewer moving part. Postgres TEXT has no practical size limit
for recruiter mail (< 1 MB per). Reporting queries stay one JOIN away.
**Revisit if:** We start ingesting attachments or hit table bloat.

## D7 — pgvector image now, extension not yet enabled (2026-08-19)
**Decision:** docker-compose runs `pgvector/pgvector:pg16`. Phase 0 does not
`CREATE EXTENSION vector` or use vector columns.
**Alternatives:** Vanilla postgres image now, swap later.
**Reason:** Image swap on a bind-mount volume is annoying. Cheaper to start
with the extension-capable image and enable when Phase 1 adds embeddings.
**Revisit if:** pgvector's tagged image lags mainline Postgres releases.

## D8 — TEST_DATABASE_URL over testcontainers (2026-08-19)
**Decision:** Tests connect to whatever `TEST_DATABASE_URL` points at.
`conftest.py` creates/drops tables via `Base.metadata`.
**Alternatives:** testcontainers-python spinning docker per session.
**Reason:** User preference — one fewer runtime dep. `docker compose up db`
already gives us the DB; a `triage_test` sibling is one createdb, wired via
the init script.
**Revisit if:** CI can't provide Postgres.

## D9 — Salary normalised to LPA + currency column (2026-08-19)
**Decision:** `ctc_min_lpa` and `ctc_max_lpa` NUMERIC(10,2), plus `currency`
TEXT default `INR`. Extractor is instructed to normalise to LPA units.
**Alternatives:** Loose NUMERIC with unit as free text.
**Reason:** Downstream comparisons (fit scoring, filtering) need numeric
comparability. Indian recruiter mail speaks LPA uniformly; convert once at
extraction time rather than parsing at query time.
**Revisit if:** Non-INR volume grows, or per-month CTC phrasings appear.

## D10 — Enums stored as TEXT + Python StrEnum, not PG native ENUM (2026-08-19)
**Decision:** `messages.status`, `messages.category`, `opportunities.work_model`,
`opportunities.employment_type` are `String` columns; Python `StrEnum`s enforce
values in-app.
**Alternatives:** PostgreSQL native ENUM types.
**Reason:** Native enums require `ALTER TYPE ... ADD VALUE` for extension,
which does not run inside a transaction on older PG and has surprising
autocommit semantics on newer PG. TEXT + StrEnum makes adding a state a
code-only change.
**Revisit if:** We need enum values to appear as first-class DB-visible types
for BI tooling.

## D11 — Rules are code, not prompt instructions (2026-08-19)
**Decision:** The five decliner rules are Python functions in
`app/rules/decliners.py`. No decline logic lives in an LLM prompt.
**Alternatives:** Encode the rules in the extractor's system prompt; encode
them in a "policy check" LLM call after extraction.
**Reason:** COST — code runs in microseconds, prompts re-pay tokens per
message. DETERMINISM — same input, byte-identical output today, next month,
next model version. TESTABILITY — one-line `assert rule(opp) == verdict`
vs statistical evaluation. AUDITABILITY — when Legal asks "why did we
decline this," we point at a function and a commit; we cannot point at a
token distribution. UNBYPASSABILITY — a prompt rule is a request that
injected content can influence; a code rule is a constraint.
**Revisit if:** A rule's logic outgrows what regex + comparators express
naturally (e.g. semantic understanding of "the client asks for a US work
authorisation" without keyword matches) — then an LLM check that *feeds*
a code rule, not one that replaces it.

## D12 — Drafts are template-only, no LLM in drafting (2026-08-19)
**Decision:** Reply drafts come from `str.format` on committed templates
(`app/drafts/templates/*.txt`). No LLM call in the drafting path.
**Alternatives:** LLM-generated drafts, gated by the outbound validator.
**Reason:** Every LLM-generated word could hallucinate a claim about the
candidate, mis-quote a salary, or absorb an injection payload extracted
into jd_text. Static templates cannot lie, cost zero, are deterministic,
and pass the outbound validator by construction because the surface area
is exactly what's written down.
**Revisit if:** Recruiter mail becomes structured enough (or drafts need
enough per-role variability) that templates start looking cramped — even
then, prefer more templates over LLM generation.

## D13 — Trust boundary: LLM nodes are tool-free and I/O-free (2026-08-19)
**Decision:** `app/pipeline/classify.py`, `app/pipeline/extract.py`, and
`app/scoring/fit.py` invoke exactly one LLM surface —
`LLMClient.structured_completion` — which has no `tools=` parameter, no
function-calling capability, no MCP surface. These modules do not import
DB session helpers, do not touch the filesystem, and do not import
`requests`. Their only observable side effect is returning typed data.
The persist node is the only DB writer.
**Alternatives:** Function-calling / tool use to let the model fetch
additional context or trigger side effects. Multiple LLM client shims
per module.
**Reason:** Structural safety. Whatever an email says — however cleverly
it says it — cannot cause an LLM call to fire an HTTP request, spawn a
shell, delete a row, or write a file, because the surface exposed to the
LLM is `str/int/enum → BaseModel`. Routing, persistence, and validation
all live in deterministic downstream code that reads the model's typed
output. This is verified by `test_llm_client_exposes_no_tool_surface`.
**Revisit if:** A phase genuinely requires the model to fetch external
context (e.g. company lookups). At that point, wrap the fetch in a
deterministic Python function invoked by an orchestrator node, NOT as
a tool the LLM can call.

## D14 — Outbound validator is a hard code gate (2026-08-19)
**Decision:** `app/drafts/validator.py` scans every draft for PAN /
Aadhaar patterns and enforces `max_length_chars` before persist. Rejection
= quarantine + reason, never silent strip.
**Alternatives:** Prompt-only "don't include PII" instruction; LLM-based
review of the draft; silent redaction with `xxxx` masks.
**Reason:** A prompt is a request. Sample the same prompt enough times
and one output slips through. Code validators are deterministic constraints
— regex matches or it doesn't. Silent redaction hides the failure; a
human reviewing "why was this quarantined" is a better outcome than a
sanitised message that looks fine and got sent.
**Revisit if:** The regex list grows past a handful of patterns — then
consider a dedicated PII library (presidio) rather than more ad-hoc regex.

## D15 — Fit threshold 65; uncertain → NEEDS_REVIEW, score ignored (2026-08-19)
**Decision:** Fit threshold defaults to 65 (in `candidate.toml [scoring]`).
`uncertain=true` from the scorer routes to `NEEDS_REVIEW` with no draft
generated, regardless of the score value. Score < threshold generates a
soft-decline draft; score >= threshold generates an interested draft.
**Alternatives:** Lower threshold (more drafts, more human review load);
treat uncertain as low-score; auto-generate all drafts and let human
approve/reject each.
**Reason:** Middling scores from an LLM asked to score are usually the
model hedging on thin input, not a real "50/50 role." The explicit
abstain signal separates "I looked at strong evidence and score is 40"
from "the input was too thin to justify a number." Structural respect for
the abstain — no draft — makes it meaningful. If a middle number were
treated as "generate a soft decline anyway," the model would learn to
guess middles instead of abstaining.
**Revisit if:** The needs-review queue grows faster than a human can
process — the fix is a better scorer prompt, not lowering the abstain
threshold.

## D16 — India detection via a maintained token whitelist (2026-08-19)
**Decision:** `app/rules/decliners.py::INDIA_TOKENS` is a hand-curated
frozenset of Indian state/city names + "india" + region shorthands. The
outside-India rule biases toward decline on ambiguity.
**Alternatives:** `pycountry` for country matching; a geocoding library;
LLM-based location classification.
**Reason:** Location strings in recruiter mail are messy ("Bangalore/Remote",
"Onsite - Bengaluru India", "London or Bangalore"). Real geocoding is
overkill for a binary "any India signal present" test. A missed match
biases toward decline, which is the safe direction for a candidate not
currently open to relocation.
**Revisit if:** Frequent false-declines on legitimate hybrid locations
show up (e.g. "Bangalore-based team, London HQ") — then augment tokens or
switch to a proper library.

## D17 — Explicit LangGraph state schema with reducers (2026-08-19)
**Decision:** `app/pipeline/state.py` is the single-source-of-truth
TypedDict for the graph's shared state. Every field is REPLACE by default;
`events` is `Annotated[list[NodeEvent], operator.add]` for APPEND.
**Alternatives:** untyped dict; Pydantic model; LangGraph's `MessagesState`;
no reducers (last-write-wins on everything).
**Reason:** Typed state means autocomplete, static-check errors on typos,
and one place to reason about who writes what. Explicit APPEND on `events`
preserves the full node-execution audit across restarts — critical for the
Phase 2 restart test and for the API's per-thread events view. GOTCHA
documented at the annotation: if `events` were REPLACE, only the last
node's event would survive; that class of bug is silent, so it earns its
own comment in the code.
**Revisit if:** state grows past ~20 fields (consider grouping into
sub-Pydantic models) or a field acquires genuinely additive semantics
that isn't a list (e.g. running sums — LangGraph reducers can be any
callable, not just `operator.add`).

## D18 — thread_id = RFC 5322 Message-ID (2026-08-19)
**Decision:** LangGraph `thread_id` (the durable conversation identity in
the checkpointer) is the same Message-ID that is the DB primary key.
**Alternatives:** UUID per thread; Gmail internal id; hash of subject+from.
**Reason:** Same argument as D2, applied to a different layer. Message-ID
is portable, RFC-stable, globally unique, and stamped at composition time.
Using it as thread_id means resume-after-restart is idempotent: whichever
process invokes `graph.invoke(..., thread_id=mid)` picks up the same
paused thread. Two processes cannot accidentally "own" the same email
because Postgres serialises checkpoint writes per thread_id.
**Revisit if:** we ingest mail from a source that lacks Message-IDs (see
D2's revisit condition — same fallback).

## D19 — Gmail scope escalation to gmail.compose (2026-08-19)
**Decision:** Phase 2 upgrades the OAuth scope from `gmail.readonly` to
`gmail.compose`. Gmail has no drafts-only scope; `compose` also grants
send permission. Our "drafts-only" guarantee is a CODE property:
`GmailClient.create_draft` is the only Gmail-mutating method; no file in
the codebase calls `users().messages().send()`.
**Alternatives:** SMTP with an app password (bypasses Gmail's threading);
write `.eml` files to disk (loses Drafts folder integration); stay on
`readonly` and paste manually (Phase 1's workflow).
**Reason:** Draft-in-Gmail is where a professional would work — the user
opens Gmail, sees drafts threaded correctly, edits and sends from there.
Scope escalation is unavoidable for that UX. The API-layer capability
(send) is broader than the code-layer usage (draft) — that gap is
guarded by review discipline plus a test (see red-team suite for the
inspection pattern applied to `LLMClient`; a similar
`test_gmail_client_exposes_no_send_surface` would be a small Phase 2
follow-up).
**Revisit if:** Google introduces a drafts-only scope (unlikely) or we
decide to ship "send after N days if no explicit reject" — then the scope
matches actual usage.

## D20 — PostgresSaver + interrupt_before for durable pause (2026-08-19)
**Decision:** Compile the graph with `checkpointer=PostgresSaver(...)`
and `interrupt_before=["act"]`. Resume across process restarts works
because LangGraph's checkpoint state is on disk in Postgres, keyed by
thread_id.
**Alternatives:** `MemorySaver` (loses on restart); `SqliteSaver`
(single-writer, blocks concurrent CLI + API); external queue (Redis,
Kafka) with polling.
**Reason:** The whole point of Phase 2 is surviving "human approves 4
hours later, possibly after a process restart." MemorySaver makes that
impossible. SqliteSaver would work for one process but not for the CLI-
plus-API deployment shape. Postgres supports concurrent transactions
across processes and we already run it. Interrupt (vs polling) means the
process is free to exit between pause and resume — no held-open
coroutines, no periodic wake-ups.
**Revisit if:** we outgrow Postgres for throughput (very unlikely for
this workload) or LangGraph deprecates PostgresSaver in favor of a
different sync API.

## D21 — persist splits into pending / final / terminal (2026-08-19)
**Decision:** Three persist nodes in Phase 2:
  - `persist_pending`  writes to DB before the interrupt (visibility for API)
  - `persist_final`    updates status after act (approve or reject terminal)
  - `persist_terminal` handles paths that never reach the interrupt
                       (skipped, extraction_failed, needs_review)
**Alternatives:** one persist at end (Phase 1 shape); persist inside the
API endpoint (couples domain writes to HTTP surface).
**Reason:** The FastAPI /pending endpoint needs a domain-shaped view of
"awaiting approval" that a database index can serve. Storing that only
in LangGraph's checkpoint tables would force the API to reverse-engineer
LangGraph's internal shape — fragile, and coupling us to a moving library
version. Split persist means the DB is authoritative for "what needs
human eyes"; the checkpointer is an implementation detail for state-
machine resume. Three nodes instead of two because the short-circuit
paths (never-drafted messages) still need to land in the DB with a
terminal status, and it's simpler to have one node for that than to
retrofit persist_pending to also handle them.
**Revisit if:** the domain model grows to the point where "awaiting
approval" isn't a single boolean status (e.g., staged approvals with
multiple reviewers) — then a proper review-queue table with its own
lifecycle would replace the status column.

## D22 — Graph over autonomous agent for Phase 2 (2026-08-19)
**Decision:** Phase 2 uses a LangGraph state machine with explicit nodes,
edges, and a human-in-the-loop interrupt. It is NOT an autonomous agent
with tool access and a re-planning loop.

**Argument for the graph (what we chose):**
  - Deterministic routing → testable, debuggable, auditable. Every branch
    is a code path that a reviewer can trace by reading `_route_after_*`
    functions.
  - Explicit state → no hidden "agent memory." Every field the pipeline
    depends on is declared in `TriageState` with a documented reducer.
  - Bounded blast radius: crash in one node doesn't corrupt the reasoning
    of others; each node has a scoped responsibility that a code review
    can bound.
  - Cost predictable: no unbounded loops. LLM is called at exactly three
    known sites (classify, extract, score); anything else is free code.
  - Compliance-friendly: every action taken on the user's Gmail is a
    code-path from `act`, gated by a human at the interrupt, with a
    persisted audit trail (events list) proving what happened.

**Honest argument for an autonomous agent:**
  - Recruiter mail is HETEROGENEOUS. Real messages combine "pitching a
    role" AND "asking scheduling questions" AND "attaching a JD PDF" in
    ways our seven-category classifier flattens into one label. An agent
    with tool access and a re-planning loop would handle the mixed cases
    naturally rather than forcing them into a rigid shape.
  - Schema drift maintenance: every time recruiter behavior shifts (e.g.
    a new ATS uses a new phrasing for C2H, or LinkedIn changes its notif
    format), we're editing rules and prompts. An agent that reasons over
    the raw content might absorb that variance without our intervention.
  - Less scaffolding: one agent loop, not seven node handlers plus a
    graph plus reducers. If the goal is to move fast on a solo-operator
    use case, that scaffolding is overhead.
  - Multi-turn conversations: if the recruiter asks a follow-up question,
    a graph shaped as a linear DAG is the wrong abstraction. An agent
    that can decide "reply with a clarifying question, wait, re-ingest
    the reply" is native to that pattern.

**Conditions under which we would switch:**
  - (a) Classifier taxonomy grows past ~12 categories with hand-tuned
        branches — we're pretending the graph is a decision tree that has
        become one.
  - (b) Manual rule-addition cadence exceeds code-merge cadence — the
        graph is turning into a lookup table.
  - (c) We start needing negotiation (multi-turn back-and-forth with the
        recruiter) — the DAG shape is the wrong primitive.
  - (d) We're staffed to review agent-produced actions continuously
        (weekly eval, red-team, retraining) rather than the current
        set-and-forget expectation.
  - (e) A specific business outcome the graph cannot deliver becomes
        important (e.g., real-time negotiation, mixed-category triage in
        a single reply).
**Revisit:** review this decision at the start of every phase. If none of
(a)-(e) are true, the graph's guarantees continue to pay their
scaffolding cost.

## D23 — jd_embedding is a nullable vector(1536) column (2026-08-19)
**Decision:** `opportunities.jd_embedding` is `vector(1536)` NULLABLE.
Dimension matches text-embedding-3-small; nullable because extractions
with no jd_text (or jd_text below `min_jd_chars`) produce no embedding.
**Alternatives:** NOT NULL with a zero-vector sentinel; a separate
`opportunity_embeddings` join table; a stringly-typed JSONB blob.
**Reason:** NULL is honest — "the source didn't state it" is the same
discipline every other Opportunity field follows, and it means the
partial HNSW index (`WHERE jd_embedding IS NOT NULL`) actually skips the
zero-signal rows. A zero-vector sentinel would silently attract false
matches for any short input. A join table adds a JOIN to every dedup
read for zero benefit at Phase 4's write volume. Dimension is
schema-enforced by pgvector — inserting a wrong-length vector raises,
which is the loud failure we want for a schema drift.
**Revisit if:** We switch to text-embedding-3-large (3072-dim) or add a
second embedding model — then column-per-model or a dedicated table.

## D24 — LLMClient.embed() inherits the D13 tool-free guarantee (2026-08-19)
**Decision:** `LLMClient.embed()` takes `(text: str) → (list[float], Usage)`
and exposes no `tools=` parameter, no function-calling, no MCP surface.
Same trust-boundary property as `structured_completion`.
**Alternatives:** Route embeddings through a generic `call_openai(...)`
helper that permits tools for other call types.
**Reason:** The whole point of D13 is that the LLM surface exposed to
email-facing modules cannot fire an HTTP call, spawn a shell, or mutate
the DB no matter what the email says. Embeddings are equally email-
facing (we're passing jd_text extracted from adversary-controlled
content), so the same structural argument applies. Verified by
`tests/red_team/test_llm_client_embed_no_tool_surface.py`, which does
the same source-inspection dance as the classify/extract test.
**Revisit if:** OpenAI ever adds function-calling to the embeddings
endpoint (nonsensical today) or we introduce a second LLM library that
bundles embeddings with tool routing.

## D25 — Embed jd_text only, not a canonical concat (2026-08-19)
**Decision:** The vector is computed from `opportunity.jd_text` alone —
NOT from a canonical string like
`f"{role_title}|{company}|{location}|{jd_text}"`.
**Alternatives:** Concat all structured fields; embed subject + body;
weighted combination via separate embeddings averaged.
**Reason:** Paraphrase detection is the specific job — "same role,
different words." jd_text carries essentially all the paraphrase-worthy
signal; the structured fields (company, location, employment_type) are
either sparse or categorical and would bias the vector toward
"this-recruiter-tends-to-format-JDs-this-way" rather than "this-role-is-
about-C#-microservices." Concatenating them muddies which axis
similarity is measuring. If two vendors send the same JD with different
`company` values (the ATS repackaging case), the concat would push them
apart; jd_text-only correctly keeps them together.
**Revisit if:** Calibration shows chronic false-positives from JDs that
paraphrase well but describe genuinely different roles (e.g. same tech
stack, different domain). Then a two-vector approach — one for JD, one
for structured metadata, combined with a per-axis threshold — is the
principled fix, not stringly-concatenating them.

## D26 — HNSW over IVFFlat for the jd_embedding index (2026-08-19)
**Decision:** `CREATE INDEX ... USING hnsw (jd_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64) WHERE jd_embedding IS NOT NULL`.
**Alternatives:** IVFFlat with `lists = sqrt(N)`; no index (seq scan
until it hurts); a Faiss/Milvus sidecar for vector search.
**Reason:** Our workload is single-row inserts as emails arrive — there
is no natural batch to train IVFFlat's k-means partition against, and
cold-start on a small table gives IVFFlat awful recall until manually
retrained. HNSW builds incrementally, has good recall at low ef_search,
and its memory cost (`m * N * dim * bytes` ≈ 16 * 10_000 * 1536 * 4 ≈
940 MB at 10k rows) is well inside a workstation's headroom. A sidecar
is overkill — pgvector inside our existing Postgres is one fewer
process. The `WITH` params are pgvector's documented defaults, kept
unless the benchmark (step 6, D30) shows otherwise.
**Revisit if:** N grows past ~100k rows (memory), or recall at a given
ef_search degrades noticeably against the labelled pair set.

## D27 — Dedup lookback window defaults to 60 days (2026-08-19)
**Decision:** `candidate.toml [dedup] lookback_days = 60`. The dedup
query filters `messages.received_at >= now() - interval '60 days'`.
**Alternatives:** 30 days; 90 days; no window at all; sliding-based-on-
recruiter (per-sender window).
**Reason:** Vendor re-pitches — the specific failure the dedup pass
targets — cluster around 4-8 week cadences (recruiter follow-up
templates default to 30-day intervals; the same role gets re-pitched by
2-3 vendors within roughly a month of the first). 60 covers the tail
of that distribution without merging legitimately-new listings of a
role that was left dormant last year. Also bounds the query cost — the
HNSW index scan is proportional to matched rows, not total rows, but
the messages JOIN benefits from the received_at index on the smaller
window.
**Revisit if:** Calibration data shows false-negatives at 60d that
would be caught at 90d (e.g. clients who re-post quarterly). Cheap to
change — it's a config field, not a schema constraint.

## D28 — Dedup flags are metadata, never auto-suppression (2026-08-19)
**Decision:** duplicate_flags rows are surfaced to the human reviewer in
the approval UI but nothing in the graph reads the table to change
routing. A flagged message follows the same rules → score → draft →
persist_pending path as an unflagged one.
**Alternatives:** Auto-decline any message flagged as a duplicate;
skip drafting for flagged messages; auto-quarantine the draft.
**Reason:** The cost asymmetry is severe. False-negative on a dedup
flag = a human sees a repeat message they've already responded to
(mildly annoying, ~5 seconds of "oh right, this one again"). False-
positive on auto-suppression = a legitimate distinct opportunity is
silently dropped and the candidate never sees it (potentially a missed
job). Even a 99% precision auto-suppressor loses good opportunities at
the tail; a human-in-the-loop flag preserves the recall benefit of the
lookup without inheriting its precision failure mode. Also: the whole
pipeline's design principle (Phase 2 approval gate, code-not-prompt
rules, no LLM in drafting) is human-final. Auto-suppression breaks
that promise for the least justified reason.
**Revisit if:** The flag → review action becomes so common that it's
the dominant human-review cost — then a "batch-approve all flagged as
duplicates of X" UI shortcut, NOT auto-suppression.

## D29 — Dedup runs pre-persist, queries only prior opportunities (2026-08-19)
**Decision:** `dedup_check` sits between `embed_jd` and `rules`,
strictly before any persist node. The query returns candidates from
`opportunities` rows that existed when the query fired — the current
message's opportunity has not yet been INSERTed, so self-match is
impossible by construction.
**Alternatives:** Run dedup after persist (post-insert of the new opp);
run dedup asynchronously from a background worker after the message
lands.
**Reason:** Pre-persist matches the intent — dedup surfaces PRIOR
context to inform the human's approval on the CURRENT message. If it
ran post-persist, the new opp would trivially self-match and the query
would need a `WHERE opportunity_id != :new_id` clause; that's not
harmful but it's schema-level defence against a shape mistake we don't
need to make. Async workers add a whole new failure mode (a message
lands with no flags yet; five seconds later flags appear — the human
either has to refresh or accept stale UI) for a workload that doesn't
need it. The synchronous pre-persist call adds one round-trip per
message; at Phase 4 volume that's negligible.
**Revisit if:** Ingest throughput grows to where the sync round-trip
per message dominates latency (thousands per minute — not our
workload) or the pipeline ever needs multi-turn dedup where flags
themselves depend on downstream decisions.

## D30 — Top-K plus threshold, not either alone (2026-08-19)
**Decision:** Dedup returns at most `max_candidates_returned = 5`
neighbours whose similarity is at least `similarity_threshold = 0.85`.
Both gates apply; either alone would misbehave at the extreme.
**Alternatives:** Threshold-only (return every match above 0.85);
top-K-only (return top 5 unconditionally, no minimum similarity).
**Reason:** Threshold-only fails on a widely-pitched role — if the same
JD arrives from ten vendors, the eleventh gets ten flags, each of which
is true but the review UI drowns in them. Top-K-only fails on a
genuinely novel role — no prior is close, but we'd still return the
five furthest-known-least-far opps and pretend they're duplicates. Both
gates combined say "flag up to K prior opps, but only those that are
actually similar." K = 5 is a UX call (the review card shows a
manageable list); threshold = 0.85 is a placeholder until the
calibration script picks a data-driven number (step 5, revisit).
**Revisit if:** Calibration overrides the threshold to something
markedly different (< 0.7 or > 0.95), or the review UI can gracefully
display more than 5 flags per message (e.g. as a collapsible list).

## D33 — Autonomy criterion: blast radius × reversibility, not model confidence (2026-08-20)
**Decision:** The agent is permitted to auto-send exactly one class of
message: a rule-fired decline where the fit scorer did not run, the
draft was not quarantined, and no duplicate flags were raised. Every
other path (interested, soft decline, needs-review, quarantined,
duplicate-flagged) stays human-gated. Routing lives in
`_route_after_validate` in `app/pipeline/graph.py`.
**Alternatives:** Grant autonomy based on model confidence (send when
the fit scorer is very sure it's a bad fit); grant autonomy on any
decline; require human review for every outbound message forever.
**Reason:** Model confidence is not a robustness criterion — a
confident LLM is often confidently wrong. The two properties that
justify autonomy are: (a) the decision was made by deterministic code
verifiable in the audit trail; (b) the action's blast radius is
tightly bounded and its reversibility is high enough that a mistake
costs seconds of the recruiter's time, not a lost opportunity. A
rule-fired decline satisfies both — the rule is a Python function, the
send is a polite "not interested" to a party who processes those every
day. An interested draft fails (b) because the recipient may
misinterpret an unedited template as a firm commitment on terms. A
soft decline (score-based) fails (a) because an LLM chose the number
that triggered it.
**Revisit if:** Templates grow rich enough that even an interested
reply is safe to send unedited (unlikely — human context is exactly
where "interested" gains value), or a specific rule class starts
producing false-declines at a measurable rate (then tighten the rule,
don't loosen the autonomy gate).

## D34 — Send scope was already granted at Phase 2; auto_send just uses it (2026-08-20)
**Decision:** `GmailClient.send_reply()` (Phase 5) calls
`users().messages().send()` under the same `gmail.compose` scope Phase
2 already required. No re-consent event, no scope change. The scope's
`send` capability had been available and unused since Phase 2.
**Alternatives:** Add a second scope (`gmail.send`) — Google doesn't
offer one distinct from what `compose` grants. Add a separate OAuth
client for the auto path — same scope, more moving parts, no
guarantee benefit. Ask the user to re-authorise as a "you're granting
send" ritual — friction with no capability change.
**Reason:** The scope-vs-usage gap was already load-bearing (D19); we
gated the capability with code discipline. Phase 5 exercises that
capability from exactly one node (`auto_send_node`) protected by the
autonomy criterion (D33) and the kill switch (D36). Adding a scope
doesn't tighten the discipline; it just makes the auth flow noisier.
**Revisit if:** Google introduces a per-capability scope split (e.g.
`gmail.send` as narrower than `gmail.compose`). Then it becomes cheap
to reduce the OAuth-layer capability to what we actually use.

## D35 — Retry policy: full jitter, retryable whitelist, single-attempt for sends (2026-08-20)
**Decision:** `@retry_external(node="...")` wraps every LLM and Gmail
method. Retryable errors (rate limit, 5xx, transport timeout) sleep
`uniform(0, min(60, 2**attempt))` and retry up to 5 attempts.
Permanent-external errors (auth, badrequest, 4xx-not-429) fail on the
first attempt as `PermanentExternalError`. Domain errors from `app.*`
pass through unchanged. `send_reply` overrides with `max_attempts=1`.
**Alternatives:** Fixed backoff (`sleep(base)` every time — thundering
herd); exponential without jitter (same herd, spread across bases);
retry everything until timeout (retries auth failures, wastes tokens);
skip retry entirely (single transient failure kills a message).
**Reason:** Full jitter is the AWS Builders' Library recommendation
for a reason — clients that failed together retry across a window
rather than at a single point, so the recovering endpoint sees a
smoothed stream instead of another spike. The whitelist rule matches
D14's outbound-validator philosophy: enumerate what's allowed rather
than trying to enumerate what isn't; unknown exceptions fail
permanent, which is the safe direction. `send_reply` is the exception
because sends are not idempotent — a retry on ambiguous outcome
(request bytes on wire, response lost) sends a duplicate reply to the
recruiter, which is worse than a dead-letter the human investigates.
**Revisit if:** Provider gives us idempotency keys for send (then we
can retry safely), or the 5-attempt policy proves too aggressive
against provider quotas (then lower `max_attempts` before touching the
jitter formula — the formula is well-studied, the attempt count isn't).

## D36 — Kill switch is a DB row read per send, never at startup (2026-08-20)
**Decision:** `is_send_halted()` executes a SELECT against
`system_flags WHERE key='sends_halted'` immediately before every
Gmail-writing call (in both `auto_send` and `act`). No caching, no
startup read, no TTL. On DB read failure, returns True (fail-safe).
**Alternatives:** Environment variable read at startup; in-memory
cached read with N-second TTL; a Redis pub/sub subscription with
event-driven cache invalidation.
**Reason:** The purpose of a kill switch is "flip it and the next send
stops." A startup-read requires restart; a TTL-cache requires waiting
up to TTL; both defeat the purpose. A DB row is the coordination
primitive we already have — one SELECT is ~1ms on a warm connection,
which is negligible next to a Gmail HTTP call. Fail-safe on read
failure is deliberate: a DB blip during a send window is exactly when
we shouldn't be autonomously acting; the alternative (fail-open —
"assume unhalted if we can't check") silently discards operator intent
during exactly the window it might matter most.
**Revisit if:** The DB round-trip becomes measurable next to send
latency (unlikely at any workstation scale), or we adopt an event bus
that can push kill-switch changes to processes with lower latency than
one poll-per-send (also unlikely at this scale).

## D37 — AUTO_SENT as a new terminal state, plus auto_actioned bool (2026-08-20)
**Decision:** New `MessageStatus.AUTO_SENT` and `DraftStatus.AUTO_SENT`
values. Draft table also gets `auto_actioned BOOLEAN NOT NULL DEFAULT
FALSE`. The redundancy is deliberate.
**Alternatives:** Reuse `SENT_TO_GMAIL_DRAFTS` with only the bool
distinguishing autonomous from approved; carry a "how it was actioned"
field on Message; skip the bool and infer from `approval_reason IS
NULL`.
**Reason:** The status column carries the primary lifecycle state; a
different status means a fundamentally different event, not the same
event with a flag. An auto-send and a human-approved-draft are
different events — one wrote to the recipient's inbox without a human
in the loop, the other put a draft in the Drafts folder for a human
to send. Report queries that don't want to see auto-sent noise filter
by status; audit queries that want "everything the agent touched"
filter by both. The bool exists on top of the status because the
digest query benefits from a fast index scan on it — cheaper than a
`status IN (...)` filter.
**Revisit if:** The status enum grows past ~10 values (then reconsider
whether some can be collapsed) or the auto_actioned bool starts
appearing in queries that could just as easily filter by status (then
drop the bool and keep the enum).

## D38 — GPT-5-mini as the default deployment; temperature becomes conditional (2026-08-21)
**Decision:** Default chat deployment is `gpt-5-mini` (Global Standard,
version 2025-08-07). `temperature` is sent only when
`AZURE_OPENAI_SUPPORTS_TEMPERATURE` is true; the GPT-5 family rejects any
non-default value with HTTP 400. Call sites keep `temperature=0.0`; the
client drops it and warns once.
**Alternatives:** Stay on gpt-4o-mini — impossible, Azure blocks new
deployments of it per-subscription and retires it 2026-10-01. Fall back to
gpt-4.1-mini — same deprecation wall, it passed its 12-month new-customer
cutoff in April 2026. Sniff `"gpt-5"` in the deployment name instead of a
config flag — breaks the moment a deployment is named anything else. Coerce
temperature to 1.0 rather than omitting it — makes a forced value look like
a deliberate one in request logs.
**Reason:** Model deprecation forced the move; the flag localises an
API-shape incompatibility to one place while leaving the determinism intent
readable at the call sites. Consequence worth naming: `temperature=0.0` was
buying reproducible classification, and reasoning models cannot offer it.
Structured Outputs still guarantees the schema, so failures here are
invisible — same email may classify differently across runs. Any future
eval harness must sample repeatedly rather than assume determinism.
**Revisit if:** A cheap temperature-supporting model becomes deployable
again (flip the flag, delete nothing), or measured classification variance
turns out to affect triage outcomes rather than just borderline categories.

## D39 — reasoning_effort left unset, and why the cost model changed (2026-08-21)
**Decision:** Do not pass `reasoning_effort` yet. Record that gpt-5-mini
bills reasoning tokens at the output rate. Measured through the real
classify node: 370 in / 27 out, $0.000147 per message. An unconstrained
throwaway prompt on the same deployment emitted 152 output tokens, so
reasoning volume tracks how tightly the prompt pins the answer — the
existing classify prompt is already doing most of that work.
**Alternatives:** Set `reasoning_effort="minimal"` on classify now; expose
it as a config setting.
**Reason:** It is a real lever — seven fixed categories need no
deliberation — but it is an optimisation, and no production cost data
exists yet to size it against. The README's "~$0.08 per 100 messages" is
now wrong on both axes (rate and token volume) and should not be trusted
until `runs.estimated_cost_usd` has real numbers in it — measured classify
alone is $0.000147/msg against the README's $0.000044 estimate, ~3.3x.
**Revisit if:** The digest shows classify/score cost dominating a run, or
per-run cost exceeds roughly $0.50 per 100 messages.

## D40 — Gmail scopes are readonly + compose, superseding D19 (2026-08-21)
**Decision:** `SCOPES = [gmail.readonly, gmail.compose]`. D19's claim that
Phase 2 "upgrades the scope from readonly to compose" was wrong: Gmail
scopes are disjoint capability sets, not a ladder. `compose` grants
draft-create and send but cannot list or get inbox messages, so replacing
readonly silently broke the entire ingest path.
**Alternatives:** `gmail.modify` alone (one scope, covers read + write) —
rejected, it also grants label mutation and trashing, which no node uses.
Keep compose-only and have ingest read mail some other way (IMAP, .eml
export) — rejected, two auth mechanisms for one mailbox.
**Reason:** Two narrow scopes state the actual requirement. The bug was
invisible for a full phase because `users().getProfile()` succeeds under
compose alone — the auth smoke test printed "Authenticated as ..." and the
403 only appeared on the next call, which reads like an auth failure rather
than a scope failure. Cost of the misdiagnosis is why the corrected
reasoning now lives in the comment block at the SCOPES definition rather
than only here.
**Revisit if:** A node ever needs to label or archive processed mail (then
`gmail.modify` replaces both, and the autonomy argument gets re-examined
because trash becomes reachable), or Google ships a genuine drafts-only
scope.

## D41 — Two DB test fixtures: rollback-isolated and commit-truncate (2026-08-21)
**Decision:** `conftest.py` keeps `db_session` (wrap in a transaction, roll
back) as the default and adds `committed_db` (commit for real, TRUNCATE
... RESTART IDENTITY CASCADE on teardown). Tests whose code-under-test opens
its own connection via `session_scope()` must use `committed_db`. Also adds
`sends_enabled`, which seeds `system_flags.sends_halted = false`.
**Alternatives:** One fixture for everything, converting all ~100 tests to
commit-truncate — rejected, it makes the fast majority pay a TRUNCATE
round-trip for isolation they don't need. Keep monkeypatching `session_scope`
per test — rejected, it is the workaround that hid these four failures for a
whole phase and it silently changes the transaction semantics under test.
**Reason:** Rollback isolation assumes a single connection. `session_scope()`
deliberately opens a fresh one per unit of work, so a `Run` seeded by a test
is invisible to the graph's persist nodes and their INSERT trips the
messages.run_id FK. That is not fixable inside the rollback fixture. The
`sends_enabled` fixture exists because `is_send_halted()` is fail-safe: a
MISSING flag row reads as HALTED (D36), so an empty test DB silently halts
every send. `test_rejected_thread_does_not_call_gmail` was passing for that
reason rather than the one it claimed.
**Revisit if:** The truncate cost becomes visible in suite runtime (then
consider per-test schemas), or `session_scope` gains a test-mode hook that
makes one connection sharable — at which point one fixture could serve both.

## D42 — max_retries=0 on the Azure client; retry_external is the sole retry authority (2026-08-21)
**Decision:** `AzureOpenAI(...)` is constructed with `max_retries=0`.
**Alternatives:** Leave the SDK default of 2 and shorten retry_external's
policy to compensate; tune the SDK's own backoff and drop retry_external for
LLM calls.
**Reason:** Two retry layers multiply. The SDK retries twice by default and
honours Azure's `Retry-After: 60` on a 429, so each of retry_external's 5
attempts became up to 3 HTTP calls with 60s waits — ~15 minutes for one
embedding, against the ~30s the documented policy implies. Observed during a
100-message ingest: the run appeared hung for 10+ minutes with no new rows.
Retry belongs to retry_external because that is the layer that dead-letters
on exhaustion and feeds the digest; the SDK layer is invisible to both.
**Revisit if:** We adopt a provider SDK whose retry handles something ours
cannot (e.g. per-region failover), in which case disable ours instead — but
never run both.

## D43 — embed_jd catches PermanentExternalError, not just LLMError (2026-08-21)
**Decision:** `make_embed_jd_node` catches `(LLMError,
PermanentExternalError)`.
**Alternatives:** Catch bare `Exception` (what dedup_check does); make
`PermanentExternalError` subclass `LLMError` so the original clause works.
**Reason:** `PermanentExternalError` derives from `Exception`, not
`LLMError`, so `except LLMError` never fired for the most likely real
failure — retry exhaustion against a 429. The exception escaped the node and
`graph.invoke()`, and ingest dead-lettered the entire message, discarding a
successful classify and extract. Both this node's docstring and graph.py's
`_route_after_extract` comment promise dedup cannot short-circuit the
pipeline; that promise was false. Not made a subclass of LLMError because
retry_external is generic across Gmail and DB calls too — forcing it under
an LLM-specific base would be wrong everywhere else.
**Revisit if:** A third failure type appears here that is neither, which
would argue for the catch-all that dedup_check already uses.

## D44 — Scrub NUL bytes at the parser boundary (2026-08-21)
**Decision:** `app/gmail/parser.scrub_text` strips 0x00 from every decoded
body and every header value. Additionally, ingest's per-message handler now
catches `Exception`, not just `PermanentExternalError`.
**Alternatives:** Validate in the Pydantic `Opportunity` model; strip in the
persist layer just before INSERT; reject the whole message as malformed.
**Reason:** A recruiter mail carrying a NUL survived parse, classify, extract
and embed, then failed at `INSERT` with `DataError: text fields cannot contain
NUL (0x00) bytes` — after four paid LLM calls, and from a persist node with no
visible link to the cause. It killed a 100-message run at message 42 because
the ingest guard only covered retry exhaustion. Cleaning untrusted input where
it enters is the same argument the codebase already makes for the LLM trust
boundary. Rejecting the message was rejected: a NUL is a broken-encoder
artefact, not a signal, and the surrounding text is a real opportunity.
**Revisit if:** Other PG-illegal sequences show up (lone surrogates from
mis-decoded UTF-16 are the likely next one), which would argue for a general
"encode-safe for PG text" pass rather than a NUL special case.

## D45 — Full autonomy: AUTO_SEND_MODE, and the four D33 gates removed (2026-08-21)
**Decision:** `AUTO_SEND_MODE` = `off` | `dry_run` | `on`, defaulting to
`dry_run`. When armed, BOTH declines and interested replies send with no human
approval. D33's gates 1, 2, 4 and 5 (rule-fired only, decline-only, no fit
score, no duplicate flag) no longer gate routing.
**Alternatives:** Keep declines-only autonomy and hold interested replies for
approval; auto-send only after a soak period of reviewed drafts. Both were put
to the operator with the objection written out; full autonomy was chosen
deliberately, and that is their call to make.
**Reason:** Recorded plainly because this reverses the project's founding
premise. What remains true: the outbound validator's quarantine verdict is
still absolute in every mode — D14 makes that a code rule, and "this text must
not leave the building" is not a preference config should override. The kill
switch (D36) is still consulted per send. Default is `dry_run` because the
failure mode is an unsendable email to a real recruiter, so arming should
require a deliberate edit rather than a default.
**Known risks accepted:** classification is non-deterministic (D38); scores
near the threshold decide sends on a one-point margin; job-board no-reply
senders currently classify as `new_role_pitch` and would be replied to.
**Revisit if:** A wrong send actually happens — the first one should trigger
`AUTO_SEND_MODE=off` plus a review of which gate would have caught it.

## D46 — Candidate profile fields optional, rendered NA (2026-08-21)
**Decision:** Every `[candidate]` field except `name` is optional. Absent
values render as the literal `NA` in both the draft template and the
fit-scorer prompt via `candidate.render()`. String fields left as `FILL-ME`
are coerced to unset. `load_profile` warns at startup listing what is unfilled.
**Alternatives:** Keep fields required and fail at boot (previous behaviour);
seed placeholder values like `total_years = 0`.
**Reason:** Placeholder values were rejected outright — `0` is not "unfilled",
it is a claim of zero years' experience, and it would be sent as one. Only an
absent line says "not answered". NA is also the honest input for the scorer,
which already has an `uncertain` instruction for missing critical fields; a
zero would have it score confidently against fiction.
**Revisit if:** Drafts start going out with NA in them under
`AUTO_SEND_MODE=on` — the fix then is for the outbound validator to quarantine
any interested draft containing NA, making it a code rule rather than a
warning nobody read.

## D47 — Reply target resolved deterministically; portal alerts never answered (2026-08-21)
**Decision:** `app/rules/reply_target.resolve_reply_target` picks the address
a reply goes to, preferring the extracted `Opportunity.recruiter_email` over
`parsed.from_email`, and returning None when neither reaches a human. None
terminates the message at `_route_after_extract` with the new status
`skipped_no_reply_target` — before embed_jd, dedup, scoring or drafting. Both
`act` and `auto_send` send to the resolved address.
**Alternatives:** Skip every message from a portal domain outright (loses the
3-in-17 genuine recruiters who reach out via a Naukri relay); reply to the
relay address and let the portal forward (opaque, expires, undeliverable
silently); let an LLM judge whether a sender is a real recruiter (a constraint
that decides whether mail is sent belongs in code, per D11).
**Reason:** Measured on the first 118 real messages: 14 senders were direct HR
addresses, 3 were portal relays whose real address the extractor had already
captured from the body, and 18 were machine-generated alert digests with
nobody behind them. Over half of extracted opportunities were unanswerable.
Ordering matters — for a relay, both fields are populated and only the
extracted one reaches a person, e.g.
`ashabWlkd2VzdGNvbnN1bHRhbnRzLm5ldA==@naukri.com` (base64 of
`midwestconsultants.net`) vs `asha@midwestconsultants.net`.
Terminating before embed/score also removes roughly half the per-message LLM
spend, on mail that could never be replied to.
**Revisit if:** A portal starts relaying without exposing the recruiter's
address anywhere in the body (then the choice is reply-to-relay or drop), or
PORTAL_DOMAINS needs updating often enough to belong in config rather than code.

## D48 — The agent emits its own signature (2026-08-21)
**Decision:** `[drafts].signature` in candidate.toml is appended to every
outbound body by `generator._with_signature`, for declines and interested
alike. Optional; absent means an unsigned body.
**Alternatives:** Keep relying on the mail client (the original design); a
separate gitignored signature.txt; put the signature in the template files.
**Reason:** The original CONCEPT in generator.py assumed the user's mail
client appends their signature. That is true only for mail composed in the
Gmail web UI. A draft created via drafts.create carries exactly the MIME body
we supply, and an auto-sent reply under D45 never touches the UI — so
autonomous replies were going out unsigned. candidate.toml over signature.txt
because it is already the gitignored home for personal data and already the
only profile file mounted into the container; template files were rejected
because they are committed to git and this is contact data.
**Revisit if:** Declines and interested replies need different sign-offs, or
the signature needs per-recruiter variation — at which point it stops being
config and becomes a template concern.

## D49 — AUTO_SEND_MODE gains `draft`: autonomous drafting, manual send (2026-08-21)
**Decision:** Fourth mode `draft`. The agent runs the full autonomous path and
calls `drafts.create` instead of `messages.send`, with no approval step. The
reply lands in Gmail Drafts; a human presses send. Recorded as
`sent_to_gmail_drafts` with `auto_actioned=true`. The kill switch does NOT
gate this mode.
**Alternatives:** A separate AUTO_DRAFT boolean alongside AUTO_SEND_MODE
(rejected — two settings that can contradict each other); a new status
distinct from sent_to_gmail_drafts (rejected — the observable state is
identical to a human-approved draft, and `auto_actioned` already records who
decided; a second status would split one real state and break existing
reports); keeping approval and having the API auto-approve (rejected — it
would mean writing an approval nobody made).
**Reason:** The operator wants evidence before granting send authority. This
mode produces exactly that: identical recipient resolution (D47), body,
signature (D48) and routing as `on`, differing only in the final API call. So
a week in `draft` is a week of real data about what `on` would have done, with
every message reversible by deleting a draft.
**Why the kill switch is bypassed here:** it means "no mail leaves this
mailbox", and a draft does not leave. Bypassing lets the switch stay armed
throughout the soak without blocking the thing the operator wants. Safe
because in this mode the send call is unreachable — no configuration makes
`draft` dispatch mail. GOTCHA: this leaves auto_send asymmetric with `act`,
which does check the switch before its own create_draft. act runs only after
a human approved one specific message, so halting mid-review means "stop what
you are doing"; here drafting is the requested steady state.
**Revisit if:** Draft volume outpaces review and the queue becomes noise, or
the asymmetry with act confuses an operator during an incident.

## D50 — persist_auto was still writing Phase 5 assumptions (2026-08-21)
**Decision:** `persist_auto` now takes `draft_type` from state rather than
hard-coding `DraftType.DECLINE`, writes `fit_score`/`fit_rationale`/
`fit_uncertain`, and stores either the draft id or the sent id in
`gmail_draft_id`.
**Alternatives:** None seriously — this was a latent bug, not a design fork.
**Reason:** Under D33 the autonomous path admitted only rule-fired declines,
so a hard-coded DECLINE and absent fit fields were correct. D45 widened the
path and this writer was not updated, so every autonomously handled
INTERESTED reply was persisted as a decline with a null score. The body in
the mailbox was right; only the record of it was wrong — which is the kind of
bug that survives indefinitely because nothing visibly breaks. The original
comment on that line predicted this exact failure ("if we ever widened to
interested-and-safe-somehow, this would need to come from state.draft_type"),
which is a point in favour of writing that kind of comment.
**Revisit if:** A third writer appears with the same shape, at which point the
Draft-row construction should be extracted into one function.

## D55 — followup_existing_thread no longer drafts (2026-08-22)
**Decision:** `EXTRACTABLE_CATEGORIES` drops `FOLLOWUP_EXISTING_THREAD`. Those
messages are still classified, extracted and stored; they stop before drafting.
**Alternatives:** Deterministic rejection-phrase detection while keeping
follow-ups drafted; a new `rejection` classifier category.
**Reason:** Measured on real mail, 4 of 5 follow-up drafts were wrong. The
category held four different situations — a CTC negotiation, a rejection
("we do not have any openings that match your skill set"), a closure
("already registered… considered a duplicate"), and a technical screening
questionnaire — against a drafting layer with two templates. Rejection
detection was rejected because it would have caught two of the four and left
the agent replying into conversations regardless. The real reason is
architectural: THE PIPELINE HAS NO THREAD HISTORY. It sees one message. A cold
inbound pitch is self-contained; a reply inside a thread the candidate started
is only meaningful against context the agent cannot see, so anything it writes
is a non-sequitur.
**Revisit if:** the pipeline gains thread history (fetch the Gmail thread and
feed prior turns to the drafter). Without that, re-enabling this reintroduces
the same failure whether the drafter is a template or an LLM.

## D56 — LLM-written reply bodies, which may never be auto-sent (2026-08-22)
**Decision:** `DRAFT_MODE=template|llm`. In `llm` mode a tool-free
`structured_completion` writes the INTERESTED body; declines stay on templates.
A new `draft_source` field records which drafter ran, and `auto_send` refuses
to call `messages.send` for `draft_source == "llm"` — it creates a Gmail draft
instead, regardless of `AUTO_SEND_MODE`. Any LLM failure falls back to the
template.
**Alternatives:** Keep D12 absolute (templates only) — rejected by the
operator, and genuinely inadequate: a template cannot answer "Have you deployed
solutions using Azure AI Search?". Let LLM bodies auto-send like templates —
rejected here; see below. Feed only the extracted Opportunity rather than the
raw body — safer, but removes the ability to answer the question that
motivated the change.
**Reason:** D12 bought three things: no hallucinated claims, no mis-quoted
salary, no steering by the email being answered. The first two are addressed in
the prompt (profile is the only source of biography; never state an unlisted
fact; never name a figure below the expectation) and by D14, which still
quarantines. The third cannot be solved by prompt wording — answering a
question requires the untrusted body in the prompt, and fencing it makes
injection produce TEXT, not certainty. So the guarantee moved: an injected
instruction can change what a draft SAYS, and a human reads every draft before
it goes anywhere. That is why the auto-send prohibition is at the Gmail call
site rather than in the router — no future routing change can bypass it.
Declines stay templated because their entire content is a deterministic rule
reason, which is exactly what we want stated.
**Revisit if:** the operator wants LLM bodies to auto-send. That is a one-line
change and a materially different risk posture — it would mean text shaped by
a stranger's email leaving the mailbox unread, under the user's name.

## D57 — The CV is attached to every reply (2026-08-22)
**Decision:** `_resume_attachment` no longer consults `resume_requested`. Every
outbound reply carries the CV. The flag is still computed and still selects the
wording ("attached as requested" vs a passing mention).
**Alternatives:** Keep attaching only on request (previous behaviour).
**Reason:** A recruiter who did not think to ask still wants it, and the
alternative is them replying "please send your CV" and waiting a cycle — the
exact round-trip this agent exists to remove. Template and LLM closings were
updated at the same time: "happy to share my CV" printed next to an actual
attachment is the clearest possible sign nobody read the mail before it went.
**Revisit if:** a recruiter objects to an unsolicited attachment, or a mail
gateway starts filtering on it.

## D58 — LLM-written bodies may now be SENT; D56 reversed (2026-08-22)
**Decision:** The guard in `auto_send` that diverted `draft_source == "llm"`
to `create_draft` is removed. With `AUTO_SEND_MODE=on`, model-written replies
are emailed to recruiters with no human review.
**Alternatives:** Keep D56 and run `DRAFT_MODE=template` so sending works on
template bodies only (recommended, declined); soak in `draft` mode for a week
first (declined).
**Reason:** The operator's decision, restated after the risk was set out in
full, and it is their call to make. Recording the trade plainly: D56 existed
because answering a recruiter's question requires their untrusted words in the
drafting prompt. Fencing makes an injected instruction produce TEXT rather than
an action — D13 still holds, the LLM surface is tool-free — and D56's whole
argument was that such text is harmless only because a human reads it first.
That reader is now gone.
**What still stands:** D14 quarantine (PAN/Aadhaar, length) routes to a human
in every mode and cannot be opened by config. D36 kill switch is checked before
every send and is now the ONLY runtime control between the model and a
recruiter's inbox — considerably more load-bearing than it was. The system
prompt's hard rules (never state an unlisted fact, never name a figure below
the expectation) are unchanged but are prompt-level, not enforced.
**Known risks accepted:** classification is non-deterministic (D38); reply
addresses for Naukri relays are DERIVED by base64 decode (D59), not stated;
nothing reviews the text before it reaches a stranger.
**Revisit if:** a wrong or embarrassing send actually happens. First action is
`python -m app.cli.halt --on`, then restore the guard — `test_llm_body_IS_sent_when_armed`
is the test to flip back.

## D59 — Naukri relay addresses are decoded, not skipped (2026-08-22)
**Decision:** `decode_naukri_relay` recovers the recruiter's real address from
the relay local part (`{name}{base64(domain)}@naukri.com`). Tried after the
body lookup, before giving up.
**Alternatives:** Keep skipping relays whose body does not name an address
(previous behaviour); reply to the relay address itself.
**Reason:** Measured over 200 real messages, 13 genuine recruiters — Northwind,
Harborlane, Stellarcorp, Crestline, Midwest Consultants, BrightPath, Stellar
Hire, Talent Bridge, QuickApply, Lakeshore, Ironvale, Redpine, Kestrel (names
anonymised) — were reachable and unanswered. The
address encodes the domain; only the decode was missing. Independent
confirmation: for one message the decoder derived
`asha@midwestconsultants.net` and the extractor had separately found the
same address in the body.
**GOTCHA that cost a rewrite:** scan for the LONGEST valid base64 suffix. The
first version walked backwards and produced `sunita.rao02bm9y@thwind.com` —
a short tail of the encoded domain decodes to something domain-shaped. A
deliverable-looking wrong address is worse than no reply, and under D58 it
would be sent.
**Revisit if:** Naukri changes the relay format, or a decoded address bounces —
a bounce is the signal that the derivation is wrong for that shape.

## D60 — One reply per recruiter, deduped on the resolved address (2026-08-24)
**Decision:** New column `drafts.reply_to_email` (migration 0006, functional
index on `lower()`). `already_replied()` does one indexed lookup after the
reply target is resolved; a hit terminates the message as
`skipped_already_replied` before embed_jd. Counts SENT mail only — an unsent
draft does not block a later reply.
**Alternatives:** Dedup on `messages.from_email` (wrong — see below); derive
the recipient at query time by re-running the resolver over history; count
drafts as replies too (declined by the operator).
**Reason:** A recruiter who mails about three roles is one person, and three
near-identical generated emails read worse than none — under
AUTO_SEND_MODE=on that happens with nobody watching. Deduping on the SENDER
would not work: `resolve_reply_target` prefers the extracted recruiter address
(D47) and can decode a Naukri relay (D59), so one human arrives as
`asha@midwestconsultants.net` and
`ashabWlkd2VzdGNvbnN1bHRhbnRzLm5ldA==@naukri.com`. Two senders, one
mailbox. Recording what we actually wrote to makes the check a lookup rather
than a re-derivation that would silently change answers as the resolver
improves.
**Fails OPEN:** a DB error returns "not replied". Failing closed would make a
transient outage indistinguishable from "already handled" and silently drop a
real recruiter's first contact. A duplicate reply is embarrassing; a dropped
one is invisible.
**Backfill:** `app/cli/backfill_reply_to.py` reconstructed all 12 pre-existing
rows exactly (the resolver is pure and both inputs are still on disk), so the
history is not empty on day one.
**Revisit if:** a recruiter legitimately needs a second reply — a genuinely
new role months later — at which point this wants a time window rather than
"ever".

## D61 — Replies give information and ask nothing; CV only on request (2026-08-24)
**Decision:** Both drafters ask the recruiter no questions. The template's
clarifications block returns empty; the LLM prompt forbids questions including
the disguised form ("let me know if…", "happy to hear more about…"). The CV is
attached only when the email asks for one, reverting D57.
**Alternatives:** Ask at most one question (the prior behaviour); keep
attaching the CV to every reply (D57).
**Reason:** Operator direction. The trade worth recording: when a JD omits the
budget the reply no longer asks, so the conversation stalls until the
recruiter volunteers it — the agent gives information and waits rather than
driving. Both drafters had to change together; leaving the template asking
while the LLM stayed silent would make a reply's tone depend on which drafter
happened to run.
**GOTCHA kept in sync:** `resume_requested` now drives BOTH the attachment and
the wording. They must agree — claiming an attachment that is not there, or
offering to send one the recipient can already see, are both immediately
visible to the reader. `test_prompt_matches_the_actual_attachment_state` pins
it.
**Revisit if:** replies start reading as passive and conversations stall on
missing details, which would argue for one question again — or for surfacing
the missing fields in the digest instead of in the reply.

## D62 — The test harness may never resolve to the production database (2026-08-25)
**Decision:** `tests/conftest.py` reads TEST_DATABASE_URL from the process
environment *or* `.env`, binds `_engine` to that URL explicitly, and routes
every URL through `_assert_not_production()`, which raises if the resolved
database name equals the one `DATABASE_URL` names.
**Alternatives:** document the "export it first" requirement more loudly
(status quo — it was already documented and still failed); drop the
`drop_all` and rely on TRUNCATE (loses schema-drift detection between runs).
**Reason:** the skip guard read `settings.test_database_url`, which
pydantic-settings fills from `.env`, while the engine connected to
`settings.database_url` — production. Only a shell-exported TEST_DATABASE_URL
reconciled them. On 2026-08-24 ~13:33 UTC a plain `pytest` therefore passed
the guard, connected to `triage`, and dropped every application table. The
watcher then failed 73 consecutive cycles on `relation "runs" does not exist`,
unnoticed for ~44 hours, and all message/draft history was lost.
**Quick fix vs systemic fix:** the quick fix is `alembic stamp base && alembic
upgrade head` to rebuild the schema. The systemic fix is this entry — one
source of truth for the test URL, plus a refusal that does not depend on
anyone remembering to export anything.
**GOTCHA it leaves behind:** a poll loop that dies on its FIRST DB write is
silent, not loud. Nothing alerted; the failure was visible only in a log
nobody was reading. Detection is still unsolved — see revisit.
**Revisit if:** we add liveness alerting (a run row older than N cycles, or a
dead-letter on repeated cycle failure), which would have caught this in
minutes rather than days.

## D63 — LangSmith tracing, with redaction in code and free text dropped (2026-08-26)
**Decision:** Tracing is opt-in via `LANGSMITH_TRACING` and off by default.
When on, both trace sources — the LangGraph node tracer and the `wrap_openai`
wrapper around the Azure SDK — are handed the SAME `Client`, whose anonymizer
is built in `app/observability/redaction.py`. Free-text fields (`body_text`,
`jd_text`, `draft_body`, `subject`, `recruiter_name`, `company`, `raw_headers`,
and LLM `content`) are replaced wholesale; emails, phones, PAN and URLs are
masked everywhere else. The redactor fails CLOSED.
**Alternatives:** (a) graph-only tracing with no LLM wrapper — cheaper, but
prompts are the email body and would be the one thing left untraced;
(b) keep body text and regex-mask identifiers inside it — far better for
debugging extraction, rejected because recruiter names, employer names and CTC
survive a regex and those are exactly what the public-repo anonymisation was
about.
**Reason:** a trace is a copy of other people's mail on a vendor's servers.
Redaction therefore lives in code with tests, not in a setting — same argument
as D11 (rules) and D14 (outbound validator): a constraint that must not be
violated is not a config value.
**Fail CLOSED, against the house style:** every other guard here fails open —
`already_replied` returns False on a DB error so a real recruiter is never
silently dropped (D60). That inverts here. Failing open means an unredacted
email body on a third party's disk, which noticing later does not undo.
**GOTCHA:** `wrap_openai` builds its own client from the ambient environment
unless passed `tracing_extra={"client": ...}`. Miss that and node state is
redacted while the raw prompt uploads verbatim in the child run — and the UI
looks correct at a glance. `app/observability/tracing.py` is the only place a
client is constructed, specifically so this cannot drift.
**Revisit if:** debugging extraction proves impossible without the JD text, in
which case (b) belongs here as a SECOND replacer chosen by an explicit
setting, never as a loosening of this one.

## D64 — Off-field roles pivot to AI/ML with a CV; Naukri always attaches (2026-08-26)
**Decision:** Three changes. (1) `candidate.stack` now reads "AI/ML & GenAI
solution design and delivery (RAG systems, LLM integration, conversational AI
on Azure)", and the interested template splits total/relevant experience into
two bullets. (2) `should_attach_resume(body, from_email)` attaches whenever the
body asks OR the sender is a naukri.com relay. (3) A new `DraftType.PIVOT`
replaces the soft decline on the fit-score path: it declines the role, states
the AI/ML focus, and attaches the CV.
**Alternatives:** pivot on every decline (rejected — see the safety note);
let the LLM write the pivot (rejected: it says one fixed thing, sends
unsupervised, and carries an attachment, so there is nothing to gain and a
fabricated qualification to lose — same argument that keeps declines on
templates); rename `resume_requested` to match its widened meaning (rejected as
churn across the graph, both send nodes and their tests for no behaviour
change; a GOTCHA in resume_request.py carries the note instead).
**Reason:** operator direction. A recruiter mailing the wrong role has a
requisition list and does not know what the candidate does now; "not a fit"
teaches them nothing, a CV plus a stated focus converts a dead thread into a
standing referral.
**SAFETY — why PIVOT is scoped to the fit-score path only:** it is the only
shape that attaches a CV unasked, and `paid_placement` / `resume_service` are
decline RULES. `_route_after_rules` sends any rule verdict straight to draft,
so a solicitation can never reach the score node where PIVOT is set. The
property comes from the graph edge, not from a check someone could forget —
`test_any_rule_fire_routes_to_draft_and_never_to_the_scorer` pins the edge and
`test_the_solicitation_rules_are_actually_in_the_rule_list` pins the premise.
C2H, CTC-floor and outside-India also stay declines: the blocker there is
commercial or geographic, so pitching AI/ML at the same recruiter is pointless.
**GOTCHA kept in sync:** the pivot template asserts outright that a CV is
attached. The score node sets `resume_requested = True` in the SAME state
update that selects PIVOT, so the wording and the MIME part read one flag and
cannot disagree — the D61 invariant, extended to a third shape.
**Side effect accepted:** `stack` also feeds the fit scorer (app/scoring/fit.py),
so scoring now favours AI/ML and marks .NET roles down. That is intended, but
it moves the meaning of the existing `fit_threshold` — the Blue Yonder Staff
Data Scientist that scored 60 and declined would likely clear the bar now.
**Revisit if:** pivots draw no replies (the shape is not earning its
attachment), or the new scoring pushes borderline roles the wrong way, which
argues for retuning `fit_threshold` rather than reverting the profile.

## D65 — C2H becomes a configurable preference, not a hard rule (2026-08-26)
**Decision:** `[rules].allow_c2h` in candidate.toml. When true, `is_c2h` is
omitted from `build_rules`, so a contract-to-hire role falls through to the
fit scorer and is judged on merit. Defaults to FALSE, so an unchanged
candidate.toml keeps the old behaviour.
**Alternatives:** delete the rule outright (rejected — the operator said "for
now", and deleting loses the rule, its reason string and its test, making the
reversal a revert instead of one word); keep it and lower its precedence
(meaningless — first-match-wins, so it either fires or it does not).
**Reason:** operator direction. C2H was the one entry in the decline list that
encodes an appetite rather than a constraint — the others are money,
geography, and two kinds of solicitation. Appetite tracks the market, so it
belongs in the profile next to `ctc_floor_lpa`, not in a Python list.
**GOTCHA — the consequence that is not obvious from the flag's name:** with
the rule gone, C2H reaches the score node, which is the only place `PIVOT` is
set (D64). So an off-field C2H role now receives a CV attachment. Intended,
but it follows from the toggle rather than being stated by it, so
`test_c2h_reaches_the_scorer_when_the_toggle_is_on` pins it.
**Still enforced:** the CTC floor. C2H postings frequently quote below it, so
this does not mean every C2H role gets a reply — `is_c2h` was simply the
first rule in the list and was shadowing the money check for these.
**Revisit if:** C2H replies produce mostly low-ball conversations, which
argues for a separate (higher) floor for C2H rather than re-arming the rule.

## D66 — Score alone routes the reply; code owns the facts block (2026-08-26)
**Decision:** Two changes to the reply path. (1) The score node no longer
special-cases `uncertain` — it routes on `score >= fit_threshold` alone, so an
uncertain low score now PIVOTs. (2) The candidate's standing facts (total and
relevant experience, CTC, notice, location) are rendered by
`generator.render_facts_block` and appended in `wrap_body`; the LLM is told the
block is added automatically and instructed not to restate it.
**Alternatives:** (1) keep the abstention branch but require score above some
floor (rejected — two thresholds where one will do, and the second has no
principled value); (2) tighten the prompt to forbid merging the two experience
bullets (rejected — this was the second formatting slip, and a prompt is a
request, not a guarantee. Same argument that moved the greeting and sign-off
into code).
**Reason — (1) is a chain of two decisions, not one bug:** D54 routed
uncertain to INTERESTED because the interested template's clarifications block
would ask for whatever the JD omitted. D61 then banned questions outright and
`_clarifications_block` began returning "". The clarifying draft stopped
clarifying and nothing revisited the branch that depended on it. Measured on
live mail: 17 of 19 `interested` replies went to roles scoring BELOW threshold,
including a QA lead role and a .NET backend role — exactly what PIVOT exists to
prevent. `uncertain` is still computed and still persisted on
`drafts.fit_uncertain`; it just no longer decides what a recruiter receives.
**Reason — (2):** the model merged "Total experience" and "Relevant
experience" onto one semicolon-joined line. The values are exact and the only
correct rendering is the one candidate.toml states; the same latitude that
merged two bullets applies to the CTC figure, which is a worse thing to have
paraphrased.
**GOTCHA:** if the model ignores the instruction and lists the facts anyway,
the reply shows a visible duplicate block. That is deliberate — a loud failure
in the first draft read, rather than a silent drift noticed months later.
**Revisit if:** pivots on uncertain scores turn out to be mostly good AI/ML
roles the scorer merely could not read, which would argue for improving the
extractor's jd_text capture rather than restoring the abstention branch.

## D67 — Screening forms are answered; other follow-ups still are not (2026-08-27)
**Decision:** `followup_existing_thread` re-enters the pipeline in exactly one
case: the body is a screening form. `rules/questionnaire.py` counts known
field labels ("Current CTC:", "Notice Period:", …) and admits the message only
at MIN_FIELDS=3 or more. It routes to a new `questionnaire` node — no extract,
no embed, no dedup, no rules, no fit score — which sets `DraftType.QUESTIONNAIRE`
and the ordered field list, then rejoins the normal pipeline at `draft`.
**Alternatives:** reopen follow-ups generally (rejected — that is D55, and it
produced non-sequiturs on negotiations, rejections and closures); ask the LLM
"is this a questionnaire?" (rejected — it gates a send, so D11 applies, and
the trigger text is written by a stranger who can assert that it is one);
answer with the full standing-facts block regardless of what was asked
(rejected — these get pasted field-by-field into an ATS, so the sender's own
order and labels turn a hunt into a copy).
**Reason:** D55's real constraint was never the category label — it was that
the agent cannot answer a question whose meaning depends on thread history it
cannot see. A form asking "Current CTC:" depends on no history at all; the
answers are standing facts. D55's own comment names the skipped screening
questionnaire as a known casualty. Observed live on 2026-08-26: a TekWissen
recruiter sent a nine-field form and it was dropped as
`skipped_wrong_category` — the easiest email in the inbox to answer, since
seven of the nine values were already in candidate.toml.
**Both conditions must hold.** Category alone reopens D55 wholesale; content
alone diverts a new role pitch that happens to list three labels away from
extraction and scoring. The negative tests pin all three cases.
**GOTCHA the real message taught us:** the form wrote
`Notice Period (If currently not working, please mention last working day):`.
A pattern demanding a colon directly after the label missed the single field
recruiters care most about. Patterns now allow a parenthetical between label
and colon. The first version was tuned on invented samples and got the only
real one wrong — which is why FORM in the tests is the anonymised original.
**Unanswerable fields are DROPPED, never guessed or rendered "NA".** "NA"
beside Expected CTC reads as evasive on the exact question being screened for;
an invented value puts words in the candidate's mouth to someone who will hold
them to it. Two new optional profile fields (`native_location`,
`reason_for_job_change`) exist so the common form can be answered in full.
**No CV attached:** a follow-up thread means they already have it.
**Revisit if:** false positives appear (a JD template listing three labels),
which argues for raising MIN_FIELDS rather than adding prose heuristics.

## D68 — One answer table for both drafters; the block follows their form

**Decision:** `render_facts_block` and `build_questionnaire` now share a single
`_answer_table` and a single `render_answer_lines`, rendering one
`Label: value` per line with no bullets, no `·`-joined pairs and no alignment
padding. On the LLM path the draft node runs `detect_fields` over the
recruiter's body and passes the result to BOTH `build_llm_reply` ("these are
already answered, do not write them") and `wrap_body` ("render exactly these,
in this order"). Three profile fields were added — `current_company`,
`interview_availability`, `remote_preference`.

**The bug:** a recruiter pitched a role AND appended a screening form. That
classifies as `new_role_pitch`, so it took the extract/score/INTERESTED path
and never reached `build_questionnaire`. The model answered the form in prose;
code then appended the fixed facts block underneath. CTC, notice and location
went out twice, in two different formats.

**Root cause, and why the prompt was not the fix:** system-prompt rule 9
("answering what they asked IS the reply") and the FORMAT rule ("never list
the standing facts") are in direct contradiction when the email *is* a list of
standing-fact questions. The model obeyed the correct half. Per the standing
rule that a constraint which must not be violated lives in code, the fix moves
the decision out of the prompt: the model is handed the literal list of labels
the appended block will contain, which is checkable rather than a judgement.

**Alternatives rejected:** (a) strengthen the FORMAT wording — leaves the
contradiction intact and stakes the outcome on which rule the model weights
more that day; (b) route pitch-plus-form messages to the questionnaire path —
throws away extraction, dedup and fit scoring for a message that really is a
role pitch; (c) strip duplicated facts from the model's prose post-hoc —
string-matching a paraphrase, and it would leave the sentence mangled.

**Why the new fields are strings, not bools:** the honest answer to "available
for interview" is often "Yes, weekday evenings". A bool rounds that into a
lie. Whatever is in candidate.toml is quoted verbatim.

**GOTCHA fixed in passing:** `_COMPILED` built patterns as `pattern + _COLON`.
`|` binds looser than concatenation, so any pattern with a top-level
alternation would have lost the colon requirement on its first branch — the
single character doing most of the discrimination in that module. Patterns are
now wrapped in their own group before the colon is appended.

**GOTCHA on formatting:** do not pad labels into aligned columns. The outbound
MIME carries a text/plain part and an HTML part that is a proportional-font
`pre-wrap` div, so padding aligns in one and comes out ragged in the other —
and the HTML part is what most recipients see.

**Revisit if:** the `fact_labels` sequence ever reaches `build_llm_reply` and
`wrap_body` differently. They are two halves of one contract and the duplicate
returns the moment they disagree.

## D69 — Gmail scopes split into required and optional (2026-09-01)

**Decision:** `SCOPES` becomes `REQUIRED_SCOPES` (`gmail.readonly`,
`gmail.compose`) plus `OPTIONAL_SCOPES` (`gmail.modify`), and
`load_credentials` loads token.json with the scopes it was GRANTED
(`from_authorized_user_file(path, None)`) rather than the scopes this code
wants. Missing required → `MissingRequiredScopeError` at client construction.
Missing optional → a warning, and the one call that needs it degrades.

**What went wrong:** adding `gmail.modify` for `mark_read` — a cosmetic
feature — took the whole agent down for three days (2026-08-29 to 09-01).
Passing the code's larger scope list into `from_authorized_user_file` put it on
the hourly token REFRESH request; Google rejected the refresh with
`invalid_scope: Bad Request`, so no credentials were produced at all. Ingest,
drafting and sending do not need `modify` and were down anyway. The scope
addition also never got a DECISIONS entry, so the "this invalidates every
token.json" consequence was recorded only in a code comment.

**Alternatives rejected:** (a) catch `RefreshError` and re-run consent
automatically — the flow needs a browser and a human, neither of which exists
in the container at 3am; (b) keep the flat list and rely on remembering to
re-consent — that is what failed; (c) make `modify` required so the mismatch
is loud — turns a cosmetic gap into a hard outage, which is the bug.

**Revisit if:** a future scope becomes genuinely required by the read or send
path. Then it moves lists, and the boot preflight starts refusing to run.

## D70 — google-auth TransportError is retryable (2026-09-01)

**Decision:** add `google.auth.exceptions.TransportError` to the retryable
whitelist in `app/retry.py`.

**Why:** it does not inherit from the builtin `ConnectionError` — the MRO is
`TransportError → GoogleAuthError → Exception` — so it fell through every
branch of `is_retryable()` into the permanent bucket and dead-lettered on
attempt 1 with no retry at all. A DNS blip on 2026-09-01 dead-lettered 57
consecutive recoverable messages in under a minute this way. The module's own
docstring anticipated exactly this ("easier to notice we should retry this new
error type"); it took an outage to notice.

**Why it is safe:** `RefreshError` — the D69 bad-scope failure — is a SIBLING
of `TransportError` under `GoogleAuthError`, not a subclass. Verified against
the installed package. So genuine auth failures still fail once, not five times.

**Revisit if:** another SDK's transport exception shows up in dead_letters with
`attempts=1`. That signature — a network error that never retried — is the tell.

## D71 — A run aborts after consecutive infra failures (2026-09-01)

**Decision:** `app/cli/ingest.py` counts CONSECUTIVE `PermanentExternalError`s
and raises `InfraOutageAborted` at 5, ending the run. Any completed message
resets the count; content-quarantine failures neither increment nor reset it.

**Why:** the per-message dead-letter guard is right when failures are
independent and exactly wrong when they are shared. During a shared outage it
converts a transient problem into permanent per-message damage at full speed,
then reports the run as succeeded. Aborting leaves the remaining ids untouched
— nothing was persisted for them — so the next cycle simply re-fetches them.

**Alternatives rejected:** abort on the first infra failure — single transient
failures are normal at this volume and runs would rarely finish.

**Revisit if:** batches grow enough that 5 is noise, or if a partial-outage
pattern (every other message failing) starts slipping past the consecutive rule.

## D72 — Pinned DNS resolvers for the agent container (2026-09-01)

**Decision:** `docker-compose.yml` sets `dns: [8.8.8.8, 1.1.1.1]` on the agent.

**Why:** Docker's embedded resolver forwards to whatever the host's DNS was at
container start. On Docker Desktop the host's resolver changes on sleep/wake,
Wi-Fi switch and VPN toggle, and the container keeps forwarding to a server it
can no longer reach — producing intermittent `NameResolutionError` on
`oauth2.googleapis.com` and the Azure endpoint while the host resolves both
fine. This is the root cause D70 and D71 make survivable.

**Alternatives rejected:** rely on retries alone — the outage windows outlast
the ~63s the backoff covers.

**Revisit if:** the agent ever needs to resolve a private/internal hostname.
Public resolvers cannot see split-horizon or corporate-internal DNS.

## D73 — Reverted to draft-only autonomy (2026-09-01)

**Decision:** `AUTO_SEND_MODE` moves from `on` to `draft`. Replies are created
as Gmail drafts in the original thread; the operator reviews and presses send.
No code change — this is the switch D45 armed, returned to the position D49
describes.

**Why:** the operator elected to put themselves back between the classifier and
the recruiter. Under `on`, 17 replies went out in the 24h window ending
2026-09-01T14:46Z (ctc_below_floor=1, outside_india_no_sponsorship=2, and 14
with no rule verdict — i.e. not rule-fired declines but the D45 full-autonomy
path, LLM-written bodies sent unread). Those are unrecoverable; email has no
unsend. Draft mode keeps identical recipient resolution, body and signature and
differs only in which Gmail API call fires, so nothing about reply quality
changes — only who presses send.

**Consequences worth knowing:** (a) the kill switch does NOT gate drafting
(D49), so halting sends no longer stops anything in this mode — use
`AUTO_SEND_MODE=off` to stop the agent producing replies at all; (b) `mark_read`
runs only on the send path, so answered mail now stays UNREAD in the inbox,
which makes unread a usable "I have not dealt with this yet" signal; (c) already
-processed messages are not re-drafted — the idempotency guards skip them, so
this affects newly arriving mail only.

**Revisit if:** a soak period in `draft` shows the drafts are consistently
sent unedited. That is the evidence D49 intended this mode to produce, and it
is the only honest argument for arming `on` again.

## D74 — One activity table, fed by the event trail we already had (2026-09-01)

**Decision:** new `agent_activity` table (migration 0007) plus `app/activity.py`.
One append-only row per notable thing the agent does, so "what is it doing" is
a SELECT instead of a request to read container logs. snake_case columns, to
match the other seven tables.

**Why now:** three incidents on 2026-09-01 — a three-day auth outage, 57
dead-lettered messages, and a silent backlog — were all present in `docker
logs` and all went unnoticed. The existing tables are nouns (a message, a run,
a failure) and answer "what is the state of X"; none answers "what happened, in
order". The digest reads DB rows, so a failure occurring before any row is
written showed up as zeros, which read as a quiet week.

**The neat part:** almost no new instrumentation. Every node already emitted a
`NodeEvent` into `TriageState["events"]` via an `operator.add` reducer, and that
trail was serialised into the LangGraph checkpoint blob and never surfaced.
`flush_events` is called once from `ingest._process_one` after `graph.invoke`
returns, which captures every node's trace without any node knowing the table
exists. Explicit `record()` calls cover what happens outside the graph: watch
startup, scope preflight, run start/finish, dead letters, D71's infra abort.

**Idempotency:** UNIQUE (message_id, node, at) + ON CONFLICT DO NOTHING, because
a flush writes the whole accumulated list and the same early events reappear in
later flushes. Chosen over tracking a high-water mark in state, which is one
more thing to get wrong across a checkpoint restore.

**Alternatives rejected:** (a) a logging.Handler mirroring Python logs into the
table — captures unstructured strings and puts a DB write on every log line;
(b) a view over the existing tables — cannot show what was never persisted;
(c) keep reading `docker logs` — the status quo that lost three days.

**Known gap:** a message resumed through the API after the approval interrupt
runs a second `graph.invoke` whose events are not flushed. In draft mode almost
nothing takes that path, so this landed without it. Wire the API resume path if
approvals become common again.

**Never breaks the pipeline:** every write is wrapped and swallowed, as in
`app/dead_letter.py`. An agent that stops processing mail because it could not
write a log line would be the worse failure.

**Revisit if:** row volume outgrows a single table (only new messages generate
rows, so this is a few hundred a day today), or if a retention policy becomes
necessary. Nothing prunes this table today, deliberately.

## D75 — Subject and sender are releasable to LangSmith, behind one flag (2026-09-02)

**Decision:** new `LANGSMITH_TRACE_IDENTIFIERS`, default false. When true, a
second replacer in `app/observability/redaction.py` passes exactly three fields
through unmasked — `subject`, `from_name`, `from_email` — and `tracing.py`
stamps them into a run name and run metadata so a message is findable in the
LangSmith list view. Everything else is unchanged in both modes.

**Why:** strict redaction (D63) is correct and made tracing useless for the job
it was turned on for. Every run carried LangGraph's default name and
`[REDACTED:free-text]` where the subject was, so "is THIS email in the right
category" was unanswerable — you could not tell the rows apart to find the one
you meant. Observability you cannot navigate is not observability.

**Why a flag and not an edit:** D63's own module comment prescribed this shape
in advance — "a second replacer selected by an explicit setting, NOT a
loosening of this one". The hard rule ("prose is not sanitisable, so prose does
not travel") is untouched and untouchable by config; the flag picks between two
audited replacers that both enforce it. `body_text`, `jd_text`, `draft_body`,
`raw_headers` and LLM `content` are dropped whole in both modes, and extracted
values (`recruiter_name`, `company`, `end_client`) are NOT released — you audit
an extraction against its source, not against other extractions.

**Discovered while building, and load-bearing:** LangSmith applies the
anonymizer to run inputs, outputs and error ONLY — verified against langsmith
0.9.8, not assumed. Metadata, tags and run names bypass it entirely. So
`trace_metadata()` enforces the policy itself by not putting a value in the
dict; there is no second net beneath it. Keys are spelled `email_*` and listed
in `IDENTIFIER_KEYS` alongside the state spellings so the two channels state
one policy, with a test pinning the pairing since nothing enforces it at runtime.

**Cost, stated plainly:** this is a privacy loosening, not a display setting.
Real recruiters' names and addresses reach a third-party SaaS verbatim. It
defaults false and belongs on only where the traces are yours alone to read.

**Revisit if:** traces become shared with anyone else, or if LangSmith gains
field-level access control that would let the release be scoped to a viewer.

## D76 — The classifier states its reason, before it states its category (2026-09-02)

**Decision:** `ClassificationResult` gains a required `reason: str` capped at
200 characters, declared FIRST in the model. It flows into
`TriageState.classify_reason`, into the classify `NodeEvent` outcome, and from
there into `agent_activity` (D74) and the LangSmith node run.

**Why:** a category alone is an assertion with no way to check it.
`not_recruitment` on a real role pitch and on a newsletter are the same two
words in a trace, so auditing a miscategorisation meant re-reading the email
and guessing what the model saw. This was the second half of the same
monitoring gap D75 addresses — D75 finds the run, D76 explains it.

**Field order is the interesting part:** strict structured output is
grammar-constrained, so the model emits properties in declaration order.
`reason` first means the justification is generated BEFORE the category it
justifies, so the label is conditioned on the reasoning rather than the
reasoning being written to defend a label already chosen. Chain-of-thought for
the price of a field. Declaring it last reads better as a schema and inverts
that causality, which is why it is not last.

**Cost:** ~50 output tokens on every classify call — the highest-volume LLM
call in the pipeline, one per message before any gate. Accepted knowingly.

**Kept an audit string:** `reason` is free text written by a model that just
read untrusted email. Nothing branches on it, nothing renders it, it never
reaches an outbound draft. If it ever becomes an input to anything it inherits
the full prompt-injection surface of the body it summarises.

**Revisit if:** classify volume makes the token cost material, or if reasons
prove uniformly generic — a reason that fits any email of its category is
worth nothing and the prompt, not the field, would be at fault.

## D77 — Free-text fields drop their whole subtree, not just their own string (2026-09-02)

**Decision:** redaction now drops a value if ANY ancestor key on its path names
a free-text carrier, via `_free_text_ancestor`. Previously only the leaf key
was tested.

**Why — a real leak in shipped strict redaction, found 2026-09-02:**
`raw_headers` holds a dict, not a string. The anonymizer descended into it and
handed the replacer the path `["parsed", "raw_headers", "From"]`, whose last
key is `From` — not in `FREE_TEXT_KEYS`. The value fell through to
`mask_patterns`, which strips the address and keeps the display name, so
`Brightpath Careers <[email]>` uploaded to LangSmith out of a field that is on
the drop list *specifically because* the module comment says it carries the
sender's display name. The guarantee was stated, tested field-by-field, and
false for the one field the comment named.

**Why the leaf is excluded from the ancestor check:** `subject` belongs to both
`FREE_TEXT_KEYS` and D75's `IDENTIFIER_KEYS`. An ancestor check that saw a
field's own key would drop it before the identifier allowlist was consulted and
make D75 inert. An ancestor admits no exception; a leaf is decided by the
ordered rules in `redact`.

**What it generalises:** the same hole opens the first time any message list,
tool-call block or structured `content` field arrives as a nested object rather
than a flat string — increasingly likely as LLM payload shapes change under us.
Nesting can no longer route around a drop.

**Revisit if:** the subtree rule proves too broad in practice — a non-prose
scalar nested under a prose key that is genuinely wanted in a trace. Nothing
like that exists today.

## D78 — A relationship() is what orders INSERTs, not a ForeignKey (2026-09-02)

**Decision:** declare the missing `Message.draft` / `Draft.message`
relationship, AND flush the parent row explicitly in `persist_pending` and
`persist_auto` before adding anything that references it. Belt and braces,
deliberately.

**The incident:** `Draft.message_id` had a `ForeignKey` and no
`relationship()`. SQLAlchemy's unit of work derives flush order from
relationships between mappers, not from raw FK columns, so it was free to emit
`INSERT INTO drafts` before `INSERT INTO messages`. Postgres rejected it on
`drafts_message_id_fkey` and rolled back the transaction — including the
`messages` row, which is why the SQL log shows that INSERT never happening at
all rather than happening in the wrong order.

**Why it cost real drafts rather than just an error:** by the time
`persist_auto` runs, `auto_send` has already created the Gmail draft. A failed
transaction left a real draft in the mailbox with no database row pointing at
it, so the next cycle's `gmail_id` and `message_id` dedup guards saw an unseen
message and drafted for it again — once every POLL_INTERVAL_MINUTES,
indefinitely. Twelve drafts for two recruiters before it was noticed, and
deleting them in Gmail did nothing because the agent never knew they existed.
`already_replied` could not help either: it counts AUTO_SENT only (D60), and
there was no draft row to count.

**Why it hid for months:** every path carrying an Opportunity calls `s.flush()`
to populate `opp_row.id` for a downstream FK, and that flush forced `messages`
out first by accident. The questionnaire path (D67) is the only route to
`persist_auto` with `opportunity is None` — the first path with no flush, and
it went straight to production behaviour.

**Why both fixes and not one:** the relationship makes ordering correct for
every call site that will ever exist; the explicit flush makes it correct at
the point a reader is actually looking, instead of correct because of a
declaration in another module. For a bug whose failure mode is silent
duplicate outbound mail, one of each was the right price.

**No migration:** this changes ORM flush order, not schema. The FK constraint
has existed in the database since migration 0001; only the ORM was unaware.

**Known and NOT fixed here:** the irreversible Gmail side effect still happens
before the durable record is written, so ANY future failure in `persist_auto`
reproduces this same loop from a different cause. Writing an intent row before
calling Gmail, or moving the send after the commit, is the real repair and is
a larger change than a hotfix should carry.

**Revisit if:** the ordering issue recurs anywhere (it would mean a new mapper
pair with a bare ForeignKey), or when the intent-row change above is taken on.
