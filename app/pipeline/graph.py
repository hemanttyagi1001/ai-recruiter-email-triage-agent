"""
LangGraph wiring — Phase 2.

Shape (Phase 5):

    START
      └→ ingest → classify ─┬─ (skip) ─→ persist_terminal → END
                            └─ (extract) → extract ─┬─ (failed) → persist_terminal → END
                                                    └─ (ok) → embed_jd → dedup_check → rules
                                                                                          │
                              ┌───────────────────────────────────────────────────────────┘
                              ▼
                            rules ─┬─ (fired) ─→ draft → validate ─┬─ (auto eligible) ─→ auto_send → persist_auto → END
                                   │                               └─ (needs human)   ─→ persist_pending → ▓INTERRUPT▓ → act → persist_final → END
                                   └─ (passed) → score ─┬─ (uncertain) → persist_terminal → END
                                                        └─ (scored)   → draft → validate → (as above)

=============================================================================
CONCEPT: what the checkpointer physically stores, and why Postgres.
=============================================================================
`PostgresSaver` creates three tables in our database:
  - checkpoints        — snapshot of state (as JSON) after each superstep,
                         keyed by (thread_id, checkpoint_ns, checkpoint_id).
  - checkpoint_writes  — the per-channel writes that produced each snapshot;
                         used for exact replay / debugging.
  - checkpoint_blobs   — larger payloads (long strings, big objects) offloaded
                         from checkpoints for efficient JSON storage.

`thread_id` is the durable conversation identity. Every checkpoint for the
same thread_id is causally related — resuming a thread means loading its
latest checkpoint and continuing from the recorded next_node.

For our pipeline: thread_id = RFC 5322 Message-ID of the recruiter email.
This inherits D2's portability + idempotency argument at a different layer
(D2 = DB primary key; D18 = LangGraph thread identity).

Why Postgres and not MemorySaver / SqliteSaver:
  - MemorySaver holds state in a process-local dict. Every paused thread
    dies with the process. Restart-survival is impossible.
  - SqliteSaver persists to disk (survives restart) but SQLite is single-
    writer at the process level — the ingest CLI and the FastAPI service
    running concurrently would block each other on writes. Postgres allows
    concurrent transactions across processes, which matches our
    "CLI and API are peers" architecture.

=============================================================================
CONCEPT: interrupt vs polling.
=============================================================================
Compile-time `interrupt_before=["act"]` means: when execution reaches the
edge INTO the act node, LangGraph writes a final checkpoint, returns from
graph.invoke() normally (no exception), and the process is free to exit.

Alternative (polling): keep the process alive after validate, spin in a
loop asking "any approval yet?" every N seconds. Wastes CPU when idle.
Loses everything on crash. Scales poorly (one held-open coroutine per
paused thread).

With interrupt + PostgresSaver, the resume path is:
    graph.invoke(Command(update={"approval_status": "approved", ...}),
                 config={"configurable": {"thread_id": mid}})

The graph loads the checkpoint by thread_id, merges the Command update
into state via reducers, and runs from `act` onward. Crucially, the
resume can happen in a COMPLETELY DIFFERENT PYTHON PROCESS from the one
that paused it. The restart test in tests/test_restart.py demonstrates
exactly this.

=============================================================================
CONCEPT: routing is deterministic Python (unchanged from Phase 1).
=============================================================================
All `_route_after_*` functions are ordinary Python that read state and
return a node name. The LLM never routes. Every branch is a code path,
reviewable, testable, provable.
"""

from __future__ import annotations

import logging

from app.candidate import CandidateProfile
from app.config import settings
from app.db.models import EXTRACTABLE_CATEGORIES, Category, DraftType
from app.dedup.nodes import make_dedup_check_node, make_embed_jd_node
from app.drafts.generator import (
    DEFAULT_FACT_LABELS,
    build_decline,
    build_interested,
    build_pivot,
    build_questionnaire,
    wrap_body,
)
from app.drafts.llm_generator import build_llm_reply
from app.drafts.validator import validate_draft
from app.gmail.client import GmailClient
from app.llm.client import LLMClient
from app.pipeline.act import make_act_node
from app.pipeline.auto_send_node import make_auto_send_node
from app.pipeline.classify import classify
from app.pipeline.extract import extract
from app.pipeline.ingest_node import ingest_node
from app.pipeline.persist import (
    persist_auto,
    persist_final,
    persist_pending,
    persist_terminal,
)
from app.pipeline.state import TriageState, make_event
from app.rules.engine import build_rules, run_rules
from app.rules.already_replied import already_replied
from app.rules.reply_target import is_auto_responder, resolve_reply_target
from app.rules.questionnaire import detect_fields, is_profile_questionnaire
from app.rules.resume_request import should_attach_resume
from app.scoring.fit import fit_score

from langgraph.graph import END, START, StateGraph

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Node factories — each closes over its dependencies (LLM, profile, gmail)
# -----------------------------------------------------------------------------


def _make_classify_node(llm: LLMClient):
    def n(state: TriageState) -> dict:
        parsed = state["parsed"]
        result, usage = classify(parsed.subject, parsed.body_text, llm)
        return {
            "category": result.category,
            "classify_usage": usage,
            "events": [make_event("classify", outcome=f"category={result.category}")],
        }
    return n


def _make_extract_node(llm: LLMClient):
    def n(state: TriageState) -> dict:
        parsed = state["parsed"]
        opp, retries, error, usage = extract(parsed.subject, parsed.body_text, llm)
        # WHY resolve the reply target here rather than in its own node: it is
        # pure, deterministic and cheap, and it needs exactly what this node
        # just produced. A separate node would add a graph edge and a
        # checkpoint round-trip to run four string comparisons.
        # TRACE: reply_to is what _route_after_extract branches on next, so
        # setting it here is what lets a portal alert terminate before paying
        # for an embedding and a fit score.
        reply_to = resolve_reply_target(parsed, opp) if opp is not None else None
        # Read from the RAW body and the RAW From header, not from anything
        # the extractor produced: both are things the sender wrote or the
        # relay stamped, and a regex over them cannot be talked out of its
        # answer the way a model could.
        # D64: Naukri relay mail attaches unconditionally — the portal's flow
        # means the ask never appears as a sentence, but the recruiter still
        # expects the CV.
        resume_requested = should_attach_resume(parsed.body_text, parsed.from_email)
        # D60: one reply per recruiter. Checked here, right after the target is
        # known and before embed_jd, so a repeat message costs classify+extract
        # and nothing else.
        replied = already_replied(reply_to)
        if replied:
            log.info(
                "already replied to %s; skipping message_id=%s before draft",
                reply_to, parsed.message_id,
            )
        # An out-of-office bouncing off our own outreach is not inbound
        # recruitment. Replying to one starts a loop with a mail server.
        if reply_to is not None and is_auto_responder(parsed.subject):
            reply_to = None
        outcome = "ok" if opp is not None else f"failed after {retries} retries"
        if opp is not None and reply_to is None:
            outcome = "ok, but no reply target"
        return {
            "opportunity": opp,
            "reply_to": reply_to,
            "resume_requested": resume_requested,
            "already_replied": replied,
            "extraction_retries": retries,
            "extraction_error": error,
            "extract_usage": usage,
            "events": [make_event("extract", outcome=outcome)],
        }
    return n


def _make_questionnaire_node():
    """Prepare a screening-form reply. No LLM, no extraction, no scoring.

    TRACE: reached only from _route_after_classify when the category is
    followup_existing_thread AND the body is a form. Sets the three things the
    draft node needs — draft_type, the fields to answer, and a reply target —
    then hands straight to draft. Nothing here costs an API call.
    """
    def n(state: TriageState) -> dict:
        parsed = state["parsed"]
        fields = detect_fields(parsed.body_text)
        # WHY resolve_reply_target with opportunity=None: there is no
        # extracted opportunity on this path, so the target can only come from
        # the From header. That is correct here — a follow-up arrives in an
        # existing thread with the recruiter's real address on it, which is
        # precisely why it needs no body-scraped fallback.
        reply_to = resolve_reply_target(parsed, None)
        if reply_to is not None and is_auto_responder(parsed.subject):
            reply_to = None
        # D60 still applies: if this address already got a reply, stop. A
        # recruiter who sends a form after we answered them already has the
        # facts.
        replied = already_replied(reply_to)
        return {
            "draft_type": DraftType.QUESTIONNAIRE,
            "questionnaire_fields": fields,
            "reply_to": reply_to,
            "already_replied": replied,
            # GOTCHA: deliberately False. A follow-up thread means the
            # recruiter already has the CV — that is how the conversation
            # started. Re-attaching it is the "offering what they can already
            # see" tell that _closing exists to avoid.
            "resume_requested": False,
            "events": [make_event(
                "questionnaire", outcome=f"form with {len(fields)} field(s)"
            )],
        }
    return n


def _route_after_questionnaire(state: TriageState) -> str:
    # No mailbox to answer, or we already answered this person: stop before
    # drafting. Same two guards the extract path applies, for the same reasons.
    if state.get("reply_to") is None or state.get("already_replied"):
        return "persist_terminal"
    if not state.get("questionnaire_fields"):
        return "persist_terminal"
    return "draft"


def _make_rules_node(profile: CandidateProfile):
    rules = build_rules(profile)

    def n(state: TriageState) -> dict:
        opp = state.get("opportunity")
        if opp is None:
            return {"rule_verdict": None,
                    "events": [make_event("rules", outcome="no opportunity")]}
        verdict = run_rules(opp, rules)
        if verdict is None:
            return {"rule_verdict": None,
                    "events": [make_event("rules", outcome="pass")]}
        return {
            "rule_verdict": verdict,
            "draft_type": DraftType.DECLINE,
            "draft_reason": verdict.reason,
            "events": [make_event("rules", outcome=f"fired: {verdict.rule_name}")],
        }
    return n


def _make_score_node(llm: LLMClient, profile: CandidateProfile):
    threshold = profile.scoring.fit_threshold

    def n(state: TriageState) -> dict:
        opp = state["opportunity"]
        result, usage = fit_score(opp, profile, llm)
        updates: dict = {
            "fit_score_result": result,
            "score_usage": usage,
        }
        # D66: routing is on the SCORE alone. `uncertain` used to short-circuit
        # to INTERESTED before the threshold was ever compared (D54), on the
        # argument that an abstention is a request for information and the
        # interested template's clarifications block would ask for what the JD
        # left out. D61 then banned questions outright and
        # `_clarifications_block` began returning "" — so the clarifying draft
        # stopped clarifying, and the branch that depended on it was never
        # revisited. Measured effect: 17 of 19 `interested` replies went to
        # roles scoring BELOW threshold, including a QA lead and a .NET backend
        # role, which is exactly what PIVOT exists to prevent.
        # WHY the flag is still computed and stored: `uncertain` remains real
        # information about the scorer's confidence — it is persisted on the
        # draft row's fit_uncertain column and read by the digest. It simply no
        # longer decides what shape of reply a recruiter receives.
        # GOTCHA: an uncertain score of 0 now pivots, which is also the safer
        # reply — it states the AI/ML focus and attaches the CV rather than
        # claiming interest in a role nobody could assess.
        if result.score >= threshold:
            updates["draft_type"] = DraftType.INTERESTED
            updates["events"] = [make_event("score", outcome=f"{result.score} → interested")]
        else:
            # D64: an off-field role is answered with a PIVOT, not a decline.
            # TRACE: reaching this branch already proves the rules PASSED —
            # _route_after_rules sends any rule-fired verdict straight to
            # draft, bypassing this node entirely. That is the whole safety
            # argument for the pivot: `paid_placement` and `resume_service`
            # are rules, so a solicitation can never arrive here, and this is
            # the only shape that attaches a CV unasked.
            updates["draft_type"] = DraftType.PIVOT
            # WHY set here rather than in the draft node: the template states
            # outright that a CV is attached, and the SAME flag drives the
            # MIME part in act/auto_send. Writing both from one place is what
            # stops the wording and the attachment disagreeing (D61) — a
            # draft node that only changed the text would produce a reply
            # claiming an enclosure that isn't there.
            updates["resume_requested"] = True
            updates["events"] = [
                make_event("score", outcome=f"{result.score} → pivot to AI/ML")
            ]
        return updates
    return n


def _make_draft_node(profile: CandidateProfile, llm: LLMClient):
    def n(state: TriageState) -> dict:
        parsed = state["parsed"]
        opp = state.get("opportunity")
        draft_type = state.get("draft_type")
        resume_attached = bool(state.get("resume_requested"))
        source = "template"
        draft_usage = None

        # CONCEPT: the LLM drafter is opt-in and only ever writes the
        #   INTERESTED shape. A decline says one sentence — "not a fit,
        #   because <deterministic rule reason>" — and there is nothing for a
        #   model to add to that except risk. The rule text is generated by
        #   code and is exactly what we want stated, so declines stay on
        #   templates in every mode.
        # TRACE: on any failure we fall through to the template below. A
        #   broken LLM call must degrade to a worse reply, never to no reply —
        #   the recruiter is waiting either way.
        if settings.draft_mode == "llm" and draft_type == DraftType.INTERESTED:
            try:
                # CONCEPT: a role pitch can also BE a screening form (D68).
                #   The questionnaire path handles a form that arrives as a
                #   bare follow-up, but recruiters routinely pitch a role and
                #   append the form to the same mail. That lands here, on the
                #   INTERESTED path, and it used to produce the standing facts
                #   twice — once as the model's prose answering the form, once
                #   as the fixed block code appends underneath.
                # TRACE: detect_fields returns their labels ordered by where
                #   each appeared in their message. The SAME list goes to both
                #   calls below: to the drafter as "these are already
                #   answered, do not write them", and to wrap_body as "render
                #   exactly these, in this order". An empty list means they
                #   asked for nothing in particular and wrap_body falls back
                #   to DEFAULT_FACT_LABELS.
                fact_labels = detect_fields(parsed.body_text)
                raw, draft_usage = build_llm_reply(
                    parsed, opp, profile, llm,
                    resume_attached=resume_attached,
                    fact_labels=fact_labels or DEFAULT_FACT_LABELS,
                )
                # The model writes only the middle. Greeting and signature come
                # from the same code the templates use, so a reply reads
                # identically whichever drafter produced it.
                body = wrap_body(raw, parsed, opp, profile, fact_labels or None)
                source = "llm"
            except Exception:
                log.exception(
                    "LLM drafting failed for message_id=%s; falling back to "
                    "the template", parsed.message_id,
                )

        if source == "template":
            if draft_type == DraftType.QUESTIONNAIRE:
                # CONCEPT: a form is answered by a table of the candidate's own
                #   facts. There is nothing for a model to contribute except a
                #   chance of restating a salary figure loosely — the same
                #   reasoning that moved the facts block into code in D66,
                #   applied to a reply that is nothing BUT facts.
                body = build_questionnaire(
                    parsed, profile, state.get("questionnaire_fields") or [], opp
                )
            elif draft_type == DraftType.PIVOT:
                # CONCEPT: the pivot stays on a template even under
                #   DRAFT_MODE=llm, and the block above enforces that by
                #   testing for INTERESTED specifically. A pivot says one
                #   fixed thing — not this role, I do AI/ML, here is my CV —
                #   and there is nothing a model can add to that except the
                #   risk of inventing a qualification, on a reply that sends
                #   unsupervised and carries an attachment. Same argument
                #   that keeps declines on templates. See D64.
                body = build_pivot(parsed, opp, profile)
            elif draft_type == DraftType.INTERESTED:
                # TRACE: the same flag that makes act/auto_send attach the PDF
                # also decides the closing sentence, so the words and the MIME
                # part can never disagree about whether a CV is enclosed.
                body = build_interested(
                    parsed, opp, profile, resume_attached=resume_attached
                )
            else:
                reason = state.get("draft_reason") or "not a fit right now"
                # WHY profile is passed to a decline, which needs nothing else
                # from it: the signature. Both outbound shapes are real mail
                # from a real person and both should be signed. See D48.
                body = build_decline(parsed, reason, opp, profile)

        updates: dict = {
            "draft_body": body,
            "draft_source": source,
            "events": [make_event(
                "draft", outcome=f"{draft_type} via {source} ({len(body)} chars)"
            )],
        }
        if draft_usage is not None:
            updates["draft_usage"] = draft_usage
        return updates
    return n


def _make_validate_node(max_length: int):
    def n(state: TriageState) -> dict:
        body = state["draft_body"]
        verdict = validate_draft(body, max_length)
        outcome = "quarantine: " + verdict.reason if verdict.quarantined else "clean"
        return {
            "draft_validation": verdict,
            "events": [make_event("validate", outcome=outcome)],
        }
    return n


# -----------------------------------------------------------------------------
# Routing functions — deterministic, LLM-free
# -----------------------------------------------------------------------------


def _route_after_ingest(state: TriageState) -> str:
    # TRACE: ingest sets `final_status` only when it hit a duplicate. On
    # the happy path final_status is unset; we proceed to classify.
    if state.get("final_status") is not None:
        return "persist_terminal"
    return "classify"


def _route_after_classify(state: TriageState) -> str:
    # TRACE: classify populated `category` (one of seven Literal values)
    # via strict JSON schema. Router compares to EXTRACTABLE_CATEGORIES
    # frozenset and returns a node name. The LLM does not route.
    cat = state.get("category")
    if cat is None:
        return "persist_terminal"
    try:
        category = Category(cat)
    except ValueError:
        return "persist_terminal"
    if category in EXTRACTABLE_CATEGORIES:
        return "extract"
    # D67: the one follow-up worth answering without thread history. BOTH
    # conditions must hold — the classifier called it a follow-up AND the body
    # is a form asking for standing facts. Category alone would reopen D55
    # wholesale; content alone would divert a new role pitch that happens to
    # list three labels away from extraction and scoring.
    # TRACE: this path skips extract, embed_jd, dedup_check, rules and score
    # entirely. There is no role being pitched, so there is nothing to extract
    # or score — the questionnaire node sets what draft needs and hands over.
    if category is Category.FOLLOWUP_EXISTING_THREAD:
        parsed = state["parsed"]
        if is_profile_questionnaire(parsed.body_text):
            return "questionnaire"
    return "persist_terminal"


def _route_after_extract(state: TriageState) -> str:
    # TRACE: extract's internal retry loop returned either an Opportunity
    # (success) or None + extraction_error (all retries exhausted). On
    # success we detour through embed_jd → dedup_check (Phase 4) before
    # hitting rules. Neither dedup node can short-circuit the pipeline:
    # embed_jd's failures leave jd_embedding unset; dedup_check then
    # skips (empty candidates); the rest of the pipeline is entirely
    # independent of both.
    if state.get("opportunity") is None:
        return "persist_terminal"
    # TRACE: added Phase 6 (D47). A message with no replyable address stops
    # HERE — before embed_jd, dedup_check, rules, score and draft. That
    # ordering is the whole point: on the observed corpus roughly half of all
    # extracted opportunities are job-board alert digests, and each one that
    # got this far would otherwise pay for an embedding and a fit-score call
    # to produce a reply that no mailbox accepts.
    if state.get("reply_to") is None:
        return "persist_terminal"
    # D60 — a recruiter we have already answered stops here too, for the same
    # reason: everything downstream costs money and produces an email nobody
    # needs.
    if state.get("already_replied"):
        return "persist_terminal"
    return "embed_jd"


def _route_after_rules(state: TriageState) -> str:
    # TRACE: if a rule fired, rules node set rule_verdict (non-None) AND
    # draft_type + draft_reason. We skip scoring and go straight to draft
    # with those fields already populated.
    if state.get("rule_verdict") is not None:
        return "draft"
    return "score"


def _route_after_score(state: TriageState) -> str:
    # TRACE: score node populated fit_score_result. If uncertain, it also
    # set needs_review=True. Uncertain skips drafting entirely and
    # persists needs_review. Otherwise score node has already set
    # draft_type (and draft_reason for soft decline) so we proceed to draft.
    fs = state.get("fit_score_result")
    # GOTCHA: `fs is None` and `fs.uncertain` used to share this branch, and
    # they are not the same failure. None means the scorer never produced a
    # result — a genuine breakdown, nothing to say. uncertain means it ran and
    # declined to commit, which under D54 is answerable with a clarifying
    # reply. Collapsing them left 9 of 10 real HR emails unanswered.
    if fs is None:
        return "persist_terminal"
    return "draft"


def _route_after_validate(state: TriageState) -> str:
    """Phase 5 — decide whether this thread qualifies for the autonomy path.

    Returns "auto_send" ONLY when every gate passes. Otherwise falls
    through to the human-approval path (persist_pending → interrupt →
    act → persist_final). See D33 for the criterion.

    The gates are all-or-nothing on purpose. A single failing gate
    (draft was quarantined, a duplicate was flagged, the fit scorer
    ran) drops the thread into human review. Autonomy is granted by
    intersection, not by any single strong signal.
    """
    draft_validation = state.get("draft_validation")

    # PHASE 6 (D45): the operator elected full autonomy — declines AND
    # interested replies send without a human. Gates 1, 2, 4 and 5 from D33
    # (rule-fired only / decline-only / no fit score / no duplicate flag) are
    # deliberately no longer consulted here. That was an explicit decision
    # made against a written objection, not an oversight; the reasoning and
    # what would reverse it are recorded in D45.
    #
    # WHAT IS STILL ABSOLUTE, and why it is the one that survived:
    #   The outbound validator's quarantine verdict. D14 makes this a CODE
    #   rule rather than a preference — it is the PAN/Aadhaar scan and the
    #   length cap. Every other gate above encoded a judgement about which
    #   decisions are safe to automate, and judgements are the operator's to
    #   make. Quarantine encodes "this specific text must not leave the
    #   building", which is not a preference and is not negotiable by config.
    if draft_validation is not None and draft_validation.quarantined:
        return "persist_pending"

    # off → the Phase 5 human-approval path, unchanged.
    # dry_run and on both enter auto_send; the node itself decides whether to
    # actually call Gmail. Routing them identically is deliberate: a dry run
    # that took a different path through the graph would be measuring the
    # wrong thing.
    if settings.auto_send_mode == "off":
        return "persist_pending"

    return "auto_send"


def _route_after_act(state: TriageState) -> str:
    # TRACE: act may have been halted by the kill switch. On halt the
    # message must stay awaiting_approval — persist_final knows how to
    # handle this via state.halted, so we route there in both cases;
    # the routing is trivial. Keeping the router as a function (not an
    # unconditional add_edge) leaves room to add branches without a
    # graph reshape later.
    return "persist_final"


# -----------------------------------------------------------------------------
# Graph builder
# -----------------------------------------------------------------------------


def build_graph(
    llm: LLMClient,
    profile: CandidateProfile,
    gmail: GmailClient,
    checkpointer,
):
    """Assemble and compile the Phase 2 graph.

    Args:
        llm: the LLMClient used by classify/extract/score.
        profile: the candidate profile (drives rule floors, score threshold,
                 draft templates).
        gmail: the Gmail client used by act. Injected so tests can pass a
               fake.
        checkpointer: a langgraph BaseCheckpointSaver (PostgresSaver in
                      production, MemorySaver in tests that don't need
                      resume across processes).
    """
    g: StateGraph = StateGraph(TriageState)

    # Nodes
    g.add_node("ingest", ingest_node)
    g.add_node("classify", _make_classify_node(llm))
    g.add_node("extract", _make_extract_node(llm))
    g.add_node("embed_jd", make_embed_jd_node(llm, profile))
    g.add_node("dedup_check", make_dedup_check_node(profile))
    g.add_node("questionnaire", _make_questionnaire_node())
    g.add_node("rules", _make_rules_node(profile))
    g.add_node("score", _make_score_node(llm, profile))
    g.add_node("draft", _make_draft_node(profile, llm))
    g.add_node("validate", _make_validate_node(profile.drafts.max_length_chars))
    g.add_node("persist_pending", persist_pending)
    g.add_node("act", make_act_node(gmail))
    g.add_node("persist_final", persist_final)
    g.add_node("persist_terminal", persist_terminal)
    # Phase 5 — autonomy path nodes.
    g.add_node("auto_send", make_auto_send_node(gmail))
    g.add_node("persist_auto", persist_auto)

    # Edges
    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", _route_after_ingest,
                            {"classify": "classify", "persist_terminal": "persist_terminal"})
    g.add_conditional_edges("classify", _route_after_classify,
                            {"extract": "extract",
                             "questionnaire": "questionnaire",
                             "persist_terminal": "persist_terminal"})
    # D67: the questionnaire path rejoins the normal pipeline at `draft`, so
    # validation, the autonomy fork and every persist path are shared. A reply
    # to a form goes through the same outbound validator as everything else.
    g.add_conditional_edges("questionnaire", _route_after_questionnaire,
                            {"draft": "draft", "persist_terminal": "persist_terminal"})
    g.add_conditional_edges("extract", _route_after_extract,
                            {"embed_jd": "embed_jd", "persist_terminal": "persist_terminal"})
    # Phase 4 dedup pass: extract → embed_jd → dedup_check → rules.
    # Both dedup edges are unconditional — the nodes themselves handle
    # the skip conditions (dedup disabled, no embedding, DB error) and
    # emit an audit event describing what happened. Keeping the routing
    # trivial means the graph shape stays legible; the failure modes
    # live in one place each (nodes.py) rather than smeared across
    # routing functions.
    g.add_edge("embed_jd", "dedup_check")
    g.add_edge("dedup_check", "rules")
    g.add_conditional_edges("rules", _route_after_rules,
                            {"draft": "draft", "score": "score"})
    g.add_conditional_edges("score", _route_after_score,
                            {"draft": "draft", "persist_terminal": "persist_terminal"})
    g.add_edge("draft", "validate")
    # Phase 5 — the autonomy fork sits here. _route_after_validate reads
    # the criterion gates and returns "auto_send" or "persist_pending".
    # See D33 for what the gates check and why.
    g.add_conditional_edges(
        "validate", _route_after_validate,
        {"auto_send": "auto_send", "persist_pending": "persist_pending"},
    )
    g.add_edge("auto_send", "persist_auto")
    g.add_edge("persist_auto", END)
    g.add_edge("persist_pending", "act")   # interrupt is applied here at compile time
    # WHY conditional_edges after act even though today it always goes
    # to persist_final: leaves a home for the "halted" branch if we
    # ever want to route it differently. Today halted still routes to
    # persist_final which handles the halted flag by leaving the
    # message awaiting_approval.
    g.add_conditional_edges(
        "act", _route_after_act,
        {"persist_final": "persist_final"},
    )
    g.add_edge("persist_final", END)
    g.add_edge("persist_terminal", END)

    # CONCEPT: `interrupt_before=["act"]` at compile time means LangGraph
    # writes a checkpoint after `persist_pending` and returns from
    # graph.invoke() without executing act. The state's next-task metadata
    # records "act" — resumption picks up from there. Combined with
    # PostgresSaver, the paused thread survives process restart.
    return g.compile(checkpointer=checkpointer, interrupt_before=["act"])
