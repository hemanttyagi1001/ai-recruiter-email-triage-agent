# Recruiter Email Triage — Architecture & Processing Flow

A reference for explaining this system out loud. Every claim here is traceable
to a file in this repo; citations are given as `path:line` so you can open the
code mid-sentence if an interviewer pushes.

---

## 1. The 60-second version

An unattended process polls a Gmail inbox every 15 minutes. For each new
message it classifies whether the mail is recruitment at all, extracts a
structured `Opportunity` from free text, embeds the job description and checks
it against everything seen before to avoid answering the same role twice,
applies deterministic filter rules, scores fit against a candidate profile,
drafts a reply, runs that reply through an outbound safety validator, and then
either creates a Gmail draft or routes the message to a human for approval.

Every message is a durable, resumable LangGraph thread keyed on its RFC 5322
Message-ID. The process can be killed mid-flight and resumes from its last
checkpoint. Re-ingesting the same mailbox is a no-op.

---

## 2. Is this "agentic" or "AI-powered"? — answer this precisely

**Short answer: it is an LLM workflow with autonomous execution, not an agent
in the model-directed sense. Say that, then explain the distinction, then
explain why the non-agentic choice was deliberate.**

The industry distinction that matters (Anthropic's *Building Effective Agents*
taxonomy, which most interviewers will be working from):

| | Workflow | Agent |
|---|---|---|
| Control flow | Predefined code paths | LLM dynamically directs its own process |
| Tool use | Orchestrated by code | LLM chooses which tools to call |
| Termination | Fixed graph reaches END | Model decides it is done |

### Where this system actually sits

**Not agentic, by design, in these respects:**

- **The graph topology is fixed Python.** All 16 nodes and every edge are
  declared in `app/pipeline/graph.py:587-657`. The model does not add,
  reorder or skip a step.
- **Every routing decision is a pure Python function** reading typed state —
  `_route_after_classify`, `_route_after_extract`, `_route_after_rules`,
  `_route_after_validate` (`app/pipeline/graph.py:512`). The LLM's output is
  an *input* to these functions; it is never the decision itself.
- **The LLM has no tools at all.** `app/llm/client.py:5-11` exposes exactly two
  methods, `structured_completion` and `embed`. Neither accepts a `tools=`
  parameter, exposes function-calling, or offers an MCP surface. Quote from
  the source: *"It cannot invoke anything."*
- **No planning, no reflection loop, no self-directed termination.** The one
  bounded retry loop is the extractor's retry-with-feedback, and its bound is
  in code, not in the model's judgement.

**Genuinely agentic in these respects:**

- **Autonomy over outcomes.** It can (and did) send email to third parties with
  no human in the loop — 17 replies on 2026-09-01 before the mode was changed.
- **Durable state and resumption.** LangGraph `PostgresSaver` checkpoints after
  every superstep; `thread_id` = Message-ID (`graph.py:19-35`).
- **Human-in-the-loop as a first-class graph construct** — a compile-time
  `interrupt_before=["act"]` that returns from `invoke()` and lets the process
  exit, rather than a polling loop (`graph.py:47-56`).
- **Self-healing infrastructure behaviour** — retry with full-jitter backoff,
  error classification, dead-lettering, an outage circuit breaker, a runtime
  kill switch.

### The answer that signals seniority

> "It's a deterministic state machine with LLM-powered steps — a workflow, not
> an agent. The model classifies, extracts, scores and writes prose, but it
> never decides what happens next and it has no tool access whatsoever. That
> was a security decision, not a simplification: the input is a stranger's
> email, which is attacker-controllable text. If the model directed control
> flow or held tools, prompt injection would escalate from 'produces bad text'
> to 'takes an action'. Keeping the LLM surface tool-free means the worst a
> successful injection achieves is a badly-worded draft that a validator still
> has to clear."

**Be ready for the pushback:** *"But the repo is called an agent."* Agree
cheerfully and reframe — the word is used in the loose industry sense of "an
autonomous background process that acts on your behalf." Then give the precise
taxonomy above. Volunteering that the name is imprecise is a much stronger move
than defending it.

**Also be ready for:** *"How would you make it agentic?"* Good answer: you
wouldn't, for this input class — but if you had to, the model-directed portion
would be quarantined to a component that never sees untrusted text, e.g. a
planner operating over already-extracted, schema-validated `Opportunity`
objects rather than over raw email bodies.

---

## 3. System context

```
┌──────────────┐         ┌─────────────────────────────────────┐
│   Gmail      │◀───────▶│  agent container (poll loop)        │
│   INBOX      │  OAuth2 │  python -m app.cli.watch            │
└──────────────┘  3 scopes└─────────────┬───────────────────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   ▼                    ▼                    ▼
          ┌─────────────────┐  ┌────────────────┐  ┌─────────────────┐
          │ Azure OpenAI    │  │  PostgreSQL    │  │  FastAPI        │
          │ gpt-5-mini      │  │  + pgvector    │  │  /pending       │
          │ text-embedding  │  │                │  │  approve/reject │
          │ -3-small        │  │  8 app tables  │  └─────────────────┘
          │ (NO tools)      │  │  3 checkpoint  │
          └─────────────────┘  └────────────────┘
```

Three processes share one database as peers: the poll loop (`watch`), the
one-shot CLI (`ingest`), and the approval API. Postgres — not SQLite — because
concurrent writers across processes must not block each other (`graph.py:37-44`).

---

## 4. The processing graph

```
START
  │
  ▼
ingest ──────────────────(not recruitment / already seen)──────▶ persist_terminal ─▶ END
  │
  ▼
classify ────(questionnaire)────▶ questionnaire ──┐
  │                                               │
  │───(not recruitment)──────────────────────────▶│ persist_terminal ─▶ END
  ▼                                               │
extract ─────(extraction failed)──────────────────┘
  │
  ▼ (ok)
embed_jd  ──▶  dedup_check  ──▶  rules
                                   │
                    ┌──────────────┴──────────────┐
              (rule fired)                  (no rule fired)
                    │                              │
                    │                              ▼
                    │                            score ───(uncertain)──▶ persist_terminal ─▶ END
                    │                              │
                    │                         (scored)
                    ▼                              ▼
                  draft ◀──────────────────────────┘
                    │
                    ▼
                 validate
                    │
        ┌───────────┴────────────┐
   (autonomy path)         (needs a human)
        │                        │
        ▼                        ▼
    auto_send              persist_pending
        │                        │
        ▼                    ▓ INTERRUPT ▓  ← graph.invoke() returns; process may exit
   persist_auto                  │
        │                        ▼  (resumed by POST /pending/{id}/approve)
        ▼                       act  ← creates the Gmail draft
       END                       │
                                 ▼
                          persist_final ─▶ END
```

Source of truth: `app/pipeline/graph.py:587-664`. The docstring at the top of
that file carries the same shape.

---

## 5. Node-by-node

| Node | Does | Decides nothing / routes on |
|---|---|---|
| `ingest` | Idempotency guards, parse MIME | already-seen → terminal |
| `classify` | LLM → `Category` + confidence | not recruitment → terminal; form → questionnaire |
| `extract` | LLM → typed `Opportunity`; retry-with-feedback on invalid output | extraction failed → terminal |
| `embed_jd` | Embedding of the JD text | unconditional |
| `dedup_check` | pgvector cosine lookup vs. prior roles | flags duplicates; never blocks |
| `questionnaire` | Answers a screening form from `candidate.toml` | rejoins at `draft` (D67) |
| `rules` | Deterministic decliners in Python | fired → straight to draft (skips scoring) |
| `score` | LLM → `FitScore(score, rationale, uncertain)` | uncertain → terminal |
| `draft` | Template render **or** LLM prose (`DRAFT_MODE`) | unconditional → validate |
| `validate` | PII scan (PAN/Aadhaar) + length cap | quarantine → human, always |
| `auto_send` | `create_draft` or `send_reply` by mode | — |
| `act` | Creates Gmail draft after human approval | — |
| `persist_*` | Writes terminal/pending/final/auto state | — |

**The deterministic rules** (`app/rules/decliners.py`): `c2h`,
`ctc_below_floor`, `outside_india_no_sponsorship`, `paid_placement`,
`resume_service`. Plus `already_replied` (don't answer the same recruiter
twice) and `resume_request` (attach the CV only if they asked).

> **Interview point.** Note the shape: *rules run before scoring, and a fired
> rule skips the LLM entirely.* That is a cost and a trust decision. A hard
> requirement ("below my salary floor") is a Python comparison, not a prompt.
> The project rule is explicit — a constraint that must not be violated lives
> in code as a validator, never in a prompt string.

---

## 6. The four reply modes

`AUTO_SEND_MODE` (`app/config.py:121`) — the single most consequential setting:

| Mode | Behaviour |
|---|---|
| `off` | Everything routes to the human approval queue |
| `dry_run` | Full pipeline runs; logs what *would* have been sent. **Default.** |
| `draft` | Creates a real Gmail draft in-thread. Nothing is sent. **Current.** |
| `on` | Sends autonomously |

`dry_run` is the default because the failure mode is irreversible — email has
no unsend — and the blast radius is the user's professional reputation. A
default that requires a deliberate edit to arm is the only defensible one.

---

## 7. Safety architecture — the part worth leading with

Four independent controls, at different layers, deliberately not collapsed
into one:

1. **Tool-free LLM surface** (`app/llm/client.py:5-11`). Structural. Prompt
   injection cannot escalate to action because there is no action to reach.
2. **Outbound validator** (`app/drafts/validator.py`). PII scan and length cap.
   Quarantine routes to a human **in every mode** — `_route_after_validate`
   checks it before it checks the mode (`graph.py:540`). Config cannot open
   this gate.
3. **Kill switch** (`app/kill_switch.py`). A DB flag read *inside* the send
   function, immediately before the Gmail call — so flipping it takes effect on
   the very next message, no restart. Storage is boring; the check *location*
   is the design.
4. **Human-in-the-loop interrupt**. A LangGraph compile-time interrupt, so a
   paused message is a durable database row rather than a held-open coroutine.

> **Sharp edge, volunteer it.** The kill switch does **not** gate draft
> creation (D49) — a draft never leaves the mailbox, so halting sends must not
> block the thing you actually want during a soak. That asymmetry is
> intentional and is exactly the kind of detail a good interviewer probes.

---

## 8. Data model

| Table | Holds |
|---|---|
| `messages` | PK = **RFC 5322 Message-ID**, not a surrogate or Gmail's id |
| `opportunities` | Extracted structured role data + JD embedding (pgvector) |
| `drafts` | Body, type, status, resolved_at, reply_to_email |
| `runs` | One row per cycle: counters, token usage, cost |
| `dead_letters` | Infra failures only — never domain failures |
| `duplicate_flags` | Dedup hits |
| `system_flags` | Kill switch |
| `agent_activity` | Append-only "what happened, in order" log |
| `checkpoints` ×3 | LangGraph's durable state |

**The Message-ID primary key is the single best design story in the schema.**
Because the PK *is* the message's globally-unique identity, re-ingesting the
same mailbox collides on insert and is skipped. Zero duplicates by
construction, not by convention. The same identity is reused as the LangGraph
`thread_id`, so idempotency at the DB layer and resumability at the graph layer
share one key.

Rejected alternatives, worth naming: `(subject, from_email)` — collapses three
legitimate follow-ups into one row; Gmail's internal id — provider-specific,
forces a migration if you ever move to Outlook.

---

## 9. Operating it

```bash
python -m app.cli.watch      # the poll loop (what the container runs)
python -m app.cli.ingest     # one cycle, then exit
python -m app.cli.digest     # last 24h summary
python -m app.cli.halt --on  # kill switch
python -m app.gmail.auth     # OAuth consent (needs a browser)
```

The activity log is the primary observability surface:

```sql
-- headline feed, per-node detail filtered out
SELECT to_char(at,'MM-DD HH24:MI') t, level, node, event, left(outcome,80)
FROM agent_activity WHERE event <> 'node_completed'
ORDER BY at DESC LIMIT 40;
```

---

## 10. Numbers you should know cold

Verified 2026-09-01 — do not round these upward in an interview.

| Metric | Value |
|---|---|
| Messages processed to date | 528 |
| Poll interval / batch cap | 15 min / 200 messages |
| Cost, 24h across 9 runs | $0.2381 |
| Test suite | 327 collected — 321 passing, 6 skipped (live red-team) |
| Models | `gpt-5-mini`, `text-embedding-3-small` |
| Architectural decisions logged | 68 entries, numbered to D74 |

Cost per message is roughly a fraction of a cent — but say "about $0.24 a day
at this volume" rather than computing a per-message figure you'd have to
defend.

---

## 11. The best production story in the repo

**A three-day silent outage, 2026-08-29 to 09-01.** Worth rehearsing because it
has a clean *quick fix vs. systemic fix* split, which is exactly what senior
interviews probe for.

**What happened.** A scope, `gmail.modify`, was added for a cosmetic feature
(marking answered mail as read). The credential loader passed the *code's*
scope list into `from_authorized_user_file()`, which put it on the hourly token
*refresh* request. Google rejected the refresh — `invalid_scope` — so no
credentials were produced at all. Reading, drafting and sending, none of which
need that scope, were down for three days. Nobody noticed, because the failure
was a log line in a container.

**Quick fix.** Delete `token.json`, re-run the consent flow.

**Systemic fixes (the interesting half):**
1. Load credentials with the scopes actually *granted*, not the scopes wanted.
   Refresh then always succeeds, and a missing scope degrades the one call that
   needs it instead of taking down the process.
2. Split scopes into **required** vs **optional** so a convenience feature can
   never again break the read path.
3. A boot-time preflight that states what the token can do.
4. An `agent_activity` table, because the real failure was that a three-day
   outage produced no artifact anyone would look at.

**The second bug found while fixing the first**, which is a better story still:
`google.auth.exceptions.TransportError` does not inherit from the builtin
`ConnectionError` — its MRO is `TransportError → GoogleAuthError → Exception`.
It therefore missed every branch of the retry classifier's whitelist, was
treated as permanent, and dead-lettered on attempt 1 with *no retry*. A DNS
blip destroyed 57 recoverable messages in under a minute. Two fixes: classify
it retryable, and add a circuit breaker so N consecutive infra failures abort
the run rather than shredding the batch at full speed.

> **Why this story lands.** It contains a whitelist that was correct in design
> and incomplete in practice, a per-item error handler that was exactly wrong
> during a shared outage, and an observability gap that hid both. Lead with the
> failure mode, not the fix.

---

## 12. Honest caveats — raise these before you are asked

- **"Agent" is a stretch.** See §2. Volunteer it.
- **Single mailbox, single user.** No multi-tenancy, no auth on the approval
  API beyond running locally. Not built for it, and saying so is better than
  being caught.
- **No LLM output evaluation harness.** There are 327 tests, but they test
  code paths, not classification/extraction *quality*. There is no labelled
  golden set and no accuracy number for the classifier. If asked "how do you
  know extraction is any good" — the honest answer is manual review of the
  activity log, and the next thing you would build is a labelled eval set.
- **Dedup thresholds are not calibrated** against a labelled set of true
  duplicates; the cosine threshold was chosen by judgement.
- **The digest reads DB rows**, so a failure occurring before any row is
  written appears as zeros — which is what let the outage stay quiet. Partly
  addressed by `agent_activity`; not fully.
- **API resume path does not write activity rows** (known gap, D74).
- **Prompt injection is mitigated structurally, not tested adversarially** at
  scale — the red-team suite exists but is skipped unless run against a live
  model.

---

## 13. Likely questions

**"Why LangGraph and not a plain function pipeline?"** Durable checkpointing
and the interrupt primitive. A paused message is a DB row that survives process
death, not a coroutine. Without that, human approval means either holding
processes open or hand-rolling a state machine and its persistence.

**"Why Postgres over SQLite for checkpoints?"** Three processes write
concurrently. SQLite is single-writer at the process level, so the CLI and API
would block each other.

**"What happens if it crashes mid-message?"** It resumes from the last
checkpoint on the next cycle. The two idempotency guards skip anything already
persisted. Interruption is survivable by construction — that is the payoff for
the checkpointer work.

**"How do you stop it emailing someone twice?"** Three layers: Message-ID
primary key, a `already_replied` rule keyed on the *resolved* reply target (not
the From header — portal relays differ), and vector dedup on the JD itself.

**"Why is the recipient not just the From address?"** Portals like Naukri relay
through opaque base64 addresses. The real recruiter mailbox is recovered from
the body and recorded on the draft, so "have we replied to this person" is a
lookup rather than a recomputation.

**"What would you do differently?"** Build the eval harness before the
autonomy. The system could send email before it could measure whether its
classifications were correct — that ordering was wrong.

---

*Companion document: `DECISIONS.md` — 68 logged decisions (numbered to D74)
with the alternatives rejected and what would make each worth revisiting.*
