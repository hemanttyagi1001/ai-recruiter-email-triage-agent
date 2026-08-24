# Project standing rules

## What this project is
A recruiter email triage agent. It reads inbound recruiter email, extracts
structured opportunity data, applies filtering rules, scores fit, drafts
replies, and routes to a human for approval before any outbound action.

This is a LEARNING project. The person reading this code is a senior engineer
with 14 years of experience, strong in .NET and Python, experienced with
LangChain in production, and deliberately learning LangGraph, evaluation
methodology, and agentic system design. The code must teach.

## Stack — do not substitute without asking
Python 3.11+, FastAPI, LangGraph, PostgreSQL with pgvector, Pydantic v2,
Docker, Gmail API, Azure OpenAI. SQLAlchemy for data access. pytest for tests.
No Streamlit, no Jupyter notebooks, no ORM alternatives, no extra frameworks.

## Working method — follow this every time
1. Before writing ANY code, present a design summary: components, data flow,
   schema shapes, and the decisions you are about to make. Stop and wait for
   my approval.
2. Only after I approve, write the code.
3. After each meaningful unit, stop and tell me what to run to see it work.
4. Never build more than the current phase asks for. If you notice something
   the next phase needs, note it and move on.

## Comment conventions — this is the most important rule
Comments must teach the reader something they could not infer from the code.
Never write comments that restate the code (`# increment counter`,
`# loop over messages`). Use these tags:

- `# CONCEPT:` — explains a general idea the reader is learning. Write it as
  if teaching, 2-5 lines. Use this for things like idempotency, checkpointing,
  cosine similarity, structured output. First occurrence only; don't repeat.
- `# WHY:` — the reason this specific line or block exists. Non-obvious
  decisions only.
- `# ALTERNATIVE:` — what else could have been done here and why it was
  rejected. Use this wherever a real fork in the road existed.
- `# GOTCHA:` — a failure mode that is not visible from reading the code.
  What breaks, under what conditions.
- `# TRACE:` — for anything non-linear (graph edges, async, retries), a short
  note on what actually happens at runtime and in what order.

Aim for roughly one teaching comment per 8-12 lines of real logic. Dense
enough to learn from, sparse enough to still read the code.

Module docstrings must open with a short "What this module does and why it
exists here" paragraph, not a one-line summary.

## Decisions log
Maintain `DECISIONS.md` at the repo root. Every time you make an architectural
choice, append an entry: the decision, the alternatives considered, the reason,
and what would make you revisit it. Keep entries short — 4-6 lines.

## Things to never do
- Never put a rule that must not be violated into a prompt string. Hard rules
  live in code as validators.
- Never give a component that reads untrusted email text any tool access.
- Never invent field values. If the source does not state something, the field
  is null, not a guess.
- Never write a mock or stub without labelling it clearly and telling me.
- Never commit secrets. Use a `.env` file and `python-dotenv`.