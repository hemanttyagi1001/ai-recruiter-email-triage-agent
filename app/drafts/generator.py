"""
Draft generator. Two templates: decline and interested. NO LLM.

CONCEPT: Drafts are template-only. Every LLM-generated word is a word that
could hallucinate a claim ("I have 20 years of Rust experience"), mis-quote
a salary, or absorb an injection payload from the extracted jd_text. Static
templates cannot lie. They cost zero. They pass the outbound validator by
construction because the surface area is exactly what's spelled out here.
The only variability is slot-filling from `candidate.toml` and the
structured Opportunity — both are trusted, typed inputs, not raw email
text. See DECISIONS.md D12.

CONCEPT: Templates end BEFORE the sign-off, and the signature is appended
here rather than written into the .txt files.
  The original reasoning was that a mail client appends the user's configured
  signature, so emitting one would duplicate it. That turned out to be true
  only for mail a human composes in the Gmail web UI. A draft created through
  drafts.create carries precisely the MIME body we hand it, and an
  autonomously sent reply through messages.send never touches the UI — so
  under D45 autonomy the templates were producing unsigned mail to recruiters.
  The signature now comes from candidate.toml (`[drafts].signature`) and is
  appended by _with_signature. Templates stay free of personal contact data,
  which is what keeps them committable to git. See D48.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from app.candidate import NA, CandidateProfile, render
from app.gmail.parser import ParsedMessage
from app.llm.schemas import Opportunity

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_DECLINE_TEMPLATE = (_TEMPLATE_DIR / "decline.txt").read_text(encoding="utf-8")
_INTERESTED_TEMPLATE = (_TEMPLATE_DIR / "interested.txt").read_text(encoding="utf-8")
_PIVOT_TEMPLATE = (_TEMPLATE_DIR / "pivot.txt").read_text(encoding="utf-8")
_QUESTIONNAIRE_TEMPLATE = (_TEMPLATE_DIR / "questionnaire.txt").read_text(encoding="utf-8")


def build_decline(
    parsed: ParsedMessage,
    reason: str,
    opp: Opportunity | None = None,
    profile: CandidateProfile | None = None,
) -> str:
    body = _DECLINE_TEMPLATE.format(
        salutation=_salutation(parsed, opp),
        about_role=_about_role_clause(opp),
        reason=reason,
    )
    # WHY profile is optional here but required on build_interested: a decline
    # needs nothing from the profile except the signature, and several tests
    # and the Phase 1 shim call this with three arguments. Defaulting to None
    # keeps those working and simply produces an unsigned body, which is the
    # pre-D48 behaviour.
    return _with_signature(body, profile)


def build_interested(
    parsed: ParsedMessage,
    opp: Opportunity,
    profile: CandidateProfile,
    resume_attached: bool = False,
) -> str:
    c = profile.candidate
    # TRACE: every slot goes through render(), so an unfilled profile field
    # becomes the literal "NA" rather than raising a TypeError on format() or
    # emitting Python's "None". The template itself is unchanged — it has no
    # idea a value was missing, which is what keeps it a dumb string.
    body = _INTERESTED_TEMPLATE.format(
        salutation=_salutation(parsed, opp),
        about_role=_about_role_clause(opp),
        total_years=render(c.total_years),
        relevant_years=render(c.relevant_years),
        stack=render(c.stack),
        current_ctc_lpa=render(c.current_ctc_lpa),
        notice_period=render(c.notice_period),
        current_location=render(c.current_location),
        preferred_location=render(c.preferred_location),
        employment_status=render(c.employment_status),
        expected_ctc=_expected_ctc(c.expected_ctc_lpa),
        budget_line=_budget_line(opp, profile),
        clarifications_block=_clarifications_block(opp, profile),
        closing=_closing(opp, profile, resume_attached),
    )
    return _with_signature(body, profile)


def build_pivot(
    parsed: ParsedMessage,
    opp: Opportunity | None,
    profile: CandidateProfile,
) -> str:
    """A redirect: not this role, but here is the work I do want, plus a CV.

    CONCEPT: why an off-field role gets a third shape rather than a decline.
      A decline ends the conversation. But a recruiter mailing about a .NET
      vacancy is a person with a requisition list, and the reason they sent
      the wrong role is that they do not know what you do now. Saying "not a
      fit" teaches them nothing; saying "not this, I do AI/ML, here is my CV"
      converts a dead thread into a standing referral.

    WHY no `reason` parameter, unlike build_decline: the reason is fixed and
    is the whole point of the shape — the role is in another field. Passing
    the fit scorer's rationale through would leak a machine's assessment of
    someone's own vacancy into mail they read.

    GOTCHA: this template hard-codes the claim that a CV is attached, so it
    must only ever be called on a path that forces the attachment. The score
    node sets `resume_requested = True` in the same state update that selects
    PIVOT, which is what makes that true. See D64 — the wording and the MIME
    part read one flag, per the D61 invariant.
    """
    c = profile.candidate
    body = _PIVOT_TEMPLATE.format(
        salutation=_salutation(parsed, opp),
        about_role=_about_role_clause(opp),
        total_years=render(c.total_years),
        relevant_years=render(c.relevant_years),
        stack=render(c.stack),
        notice_period=render(c.notice_period),
        current_location=render(c.current_location),
        preferred_location=render(c.preferred_location),
    )
    return _with_signature(body, profile)


def build_questionnaire(
    parsed: ParsedMessage,
    profile: CandidateProfile,
    fields: list[str],
    opp: Opportunity | None = None,
) -> str:
    """Answer a screening form, echoing only the fields it actually asked for.

    CONCEPT: answer their form, not ours (D67).
      The obvious implementation dumps the whole standing-facts block and lets
      the recruiter find what they wanted. That is worse than it sounds: these
      forms get pasted field-by-field into an ATS, so a reply in the sender's
      own order and wording is a copy job, while a reply in ours is a hunt.
      `fields` arrives from `questionnaire.detect_fields` already ordered by
      where each label appeared in their message.

    GOTCHA: a field we hold no value for is DROPPED, never guessed and never
      rendered as "NA". Two failure modes are being avoided at once — writing
      "NA" next to "Expected CTC" reads as evasive on the exact question they
      are screening for, and inventing a plausible value puts words in the
      candidate's mouth to someone who will hold them to it. Dropping means
      the reply is visibly incomplete, which is honest and is the one outcome
      a human can still fix.
    """
    lines = render_answer_lines(profile, fields)
    if not lines:
        # Nothing answerable. The caller should not have routed here, but a
        # reply consisting of a greeting and an empty list is worse than none.
        raise ValueError("questionnaire draft has no answerable fields")
    body = _QUESTIONNAIRE_TEMPLATE.format(
        salutation=_salutation(parsed, opp),
        answers_block="\n".join(lines),
    )
    return _with_signature(body, profile)


def _answer_table(profile: CandidateProfile) -> dict[str, str]:
    """Every canonical form label mapped to the candidate's answer for it.

    One table, both drafters. Before D68 the questionnaire path and the LLM
    path each had their own renderer, which is how a single reply came to
    state the CTC twice in two different formats.
    """
    c = profile.candidate
    return {
        "Current Location": render(c.current_location),
        "Native Location": render(c.native_location),
        "Total Experience": f"{render(c.total_years)} years",
        "Relevant Experience": (
            f"{render(c.relevant_years)} years in {render(c.stack)}"
        ),
        "Preferred Location": render(c.preferred_location),
        "Reason for Job Change": render(c.reason_for_job_change),
        "Notice Period": render(c.notice_period),
        "Current CTC": f"{render(c.current_ctc_lpa)} LPA",
        "Expected CTC": _expected_ctc(c.expected_ctc_lpa),
        # WHY this maps to notice period: "last working day" is what the form
        # asks when it assumes you have already left. Answering with the
        # notice period is the truthful response for someone still employed,
        # and `employment_status` is what makes that unambiguous — so it is
        # appended rather than substituted.
        "Last Working Day": (
            f"{render(c.employment_status)} — {render(c.notice_period)} notice"
        ),
        "Current Company": render(c.current_company),
        "Available for Interview": render(c.interview_availability),
        "Interested in Remote": render(c.remote_preference),
    }


# GOTCHA: a label the detector can produce with no entry in the answer table
# renders nothing, and a table entry no pattern can produce is dead code.
# Neither is visible at runtime — the reply is just quietly missing a line —
# which is why test_questionnaire reconciles the two vocabularies directly
# against `questionnaire.FIELD_LABELS` rather than trusting them to stay in
# step. Adding a pattern there without an answer here is the failure it
# catches.

# The subset rendered when the recruiter did NOT send a form, and the order
# they appear in. These are the facts worth volunteering unasked; the rest of
# the table (native location, reason for change, last working day) is only
# offered when something explicitly asks.
DEFAULT_FACT_LABELS: tuple[str, ...] = (
    "Total Experience",
    "Relevant Experience",
    "Current Location",
    "Preferred Location",
    "Current CTC",
    "Expected CTC",
    "Notice Period",
)


def render_answer_lines(
    profile: CandidateProfile, labels: Sequence[str]
) -> list[str]:
    """Render `labels` as "Label: value" lines, dropping the unanswerable.

    CONCEPT: why code owns this and the model does not (D66).
      These values are exact, and the only correct rendering of them is the
      one candidate.toml states. A model asked to "list the facts" will
      paraphrase, reorder, merge two bullets with a semicolon, or round a CTC
      — all of which it did. The observed failure was total and relevant
      experience arriving as one run-on line, but the same freedom applies to
      the salary figure, which is a worse thing to have restated loosely.

    CONCEPT: one field per line, and no alignment padding (D68).
      The previous block packed two fields onto one line — "Current CTC: 26
      LPA · Expected: 36 LPA" — which is fine to read and wrong to use. These
      answers get pasted into an ATS one field at a time, so a line holding
      two of them has to be split by hand every single time.
    GOTCHA: do not be tempted to pad the labels into aligned columns. The
      outbound MIME carries a text/plain part AND an HTML part that is a
      proportional-font `pre-wrap` div (gmail/client.py). Space padding aligns
      in the plain part and comes out ragged in the HTML one — which is the
      part most recipients actually see.

    GOTCHA: a label with no value is DROPPED, never rendered as "NA". Rule 2
      of the LLM system prompt tells the model to omit unknown values rather
      than write the placeholder; a code-rendered block that ignored its own
      rule would look worse than the model output it replaced.
    """
    table = _answer_table(profile)
    lines = []
    for label in labels:
        value = table.get(label)
        if value is None or NA in value:
            continue
        lines.append(f"{label}: {value}")
    return lines


def render_facts_block(
    profile: CandidateProfile | None, labels: Sequence[str] | None = None
) -> str:
    """The standing-facts block appended below an LLM-written reply.

    WHY `labels` is a parameter rather than a constant (D68): when the
    recruiter's email is itself a screening form, the right block is the
    answers to THEIR fields in THEIR order — the same thing the questionnaire
    path already builds. Passing None means they asked nothing in particular,
    so the default subset is volunteered instead.

    WHY this belongs next to _with_signature rather than in llm_generator:
      it is the same category of thing. The greeting and sign-off already
      moved into code because a model writing its own would eventually invent
      a phone number. The facts block is that argument applied to the numbers.
      Both drafters now emit byte-identical blocks.

    GOTCHA: renders "" when the profile is absent, so the LLM path degrades to
    prose-only rather than raising. A reply missing its list is worse than one
    that has it and still far better than no reply.
    """
    if profile is None:
        return ""
    return "\n".join(
        render_answer_lines(profile, labels or DEFAULT_FACT_LABELS)
    )


def wrap_body(
    body: str,
    parsed: ParsedMessage,
    opp: Opportunity | None,
    profile: CandidateProfile | None,
    fact_labels: Sequence[str] | None = None,
) -> str:
    """Put a greeting on the front, the facts and signature on the back.

    WHY this is shared rather than duplicated in the LLM drafter: the greeting
    and sign-off are the candidate's identity, not the message's content, and
    they must be byte-identical whichever drafter ran. The LLM is explicitly
    told to omit both — a model writing its own "Hi X," would guess the name
    from the untrusted body instead of using the name-resolution logic tuned
    against 22 real senders, and a model writing its own sign-off would
    eventually invent a phone number.
    GOTCHA: the first LLM-written drafts shipped without either, because this
    step existed only inside the template builders. The body looked fine in
    isolation and wrong in the mailbox.

    TRACE (D66): the facts block is appended HERE, after the model's prose and
    before the signature. The model is told the block is added automatically
    and instructed not to restate those values, so the assembled reply reads
    as: greeting → the model's answer to what they asked → the answer block →
    signature. If the model disobeys and lists them anyway the result is a
    visible duplicate, which is exactly the kind of failure that shows up in
    the first draft you read rather than silently months later.

    TRACE (D68): `fact_labels` is what the recruiter's own form asked for,
    detected by `questionnaire.detect_fields` and passed down from the draft
    node. It changes WHICH facts the block carries and in what order — their
    fields in their order when they sent a form, the default subset when they
    did not. The duplicate this fixes was not the model misbehaving: a form
    email put rules 9 and FORMAT of the system prompt in direct conflict, and
    answering the form was the correct half to obey.
    """
    greeting = f"Hi {_salutation(parsed, opp)},"
    facts = render_facts_block(profile, fact_labels)
    parts = [greeting, "", body.strip("\n")]
    if facts:
        parts += ["", facts]
    return _with_signature("\n".join(parts), profile)


# --- helpers ---


def _with_signature(body: str, profile: CandidateProfile | None) -> str:
    """Append the configured signature, or return the body unchanged.

    WHY exactly one blank line between body and signature: both templates end
    with a newline and no trailing blank, so joining with "\\n" yields one
    empty line — the ordinary spacing before a sign-off. Normalising the
    signature's own leading and trailing whitespace means the result does not
    depend on whether the TOML value happened to start with a newline, which
    is the kind of thing that varies every time someone re-pastes it.
    """
    if profile is None or not profile.drafts.signature:
        return body
    return body.rstrip("\n") + "\n\n" + profile.drafts.signature.strip("\n") + "\n"


# Words that mark a display name as an organisation rather than a person.
# GOTCHA: matched against the WHOLE display name, not the extracted first
# token. "DEEPA-CRESTLINE STAFFING" contains a marker but its first token is a
# real given name, so the marker alone must not disqualify it — it only
# disqualifies a name we have no extracted recruiter to corroborate.
_ORG_MARKERS: frozenset[str] = frozenset({
    "solutions", "consulting", "consultants", "consultancy", "staffing",
    "technologies", "technology", "services", "systems", "software",
    "associates", "group", "recruitment", "recruiters", "resourcing",
    "talent", "careers", "hiring", "jobs", "alerts", "private", "limited",
    "ltd", "inc", "llc", "llp", "pvt", "corp", "company", "team", "infotech",
})

# Tokens that are never a person's given name, even standing alone.
_NON_NAME_TOKENS: frozenset[str] = frozenset({
    "hr", "hrd", "the", "team", "careers", "career", "recruitment",
    "recruiter", "recruiting", "talent", "jobs", "job", "info", "admin",
    "support", "contact", "noreply", "no", "alerts", "alert", "notifications",
    "hiring", "staffing", "manager", "mailer", "mail", "resume", "resumes",
})


def _first_name(raw: str | None) -> str | None:
    """Extract a usable given name from a display name, or None.

    CONCEPT: a salutation is the first thing the recruiter reads, and it is the
    cheapest possible tell that mail was machine-generated. "Hi Northwind," or
    "Hi DEEPA-CRESTLINE," reads as a bot in a way the rest of a correct,
    well-formatted email does not. So the bar here is not "usually right" — a
    wrong first name is worse than no name, which is why every uncertain case
    resolves to None and lets the caller fall through to "there".
    """
    if not raw:
        return None
    name = raw.strip()
    # An address in the display-name slot is a misconfigured sender, not a
    # person. "Careers@Harborlane" must not become "Hi Careers@Harborlane,".
    if "@" in name:
        return None

    # WHY split on more than whitespace: senders arrive as "Meghna.R" and
    # "DEEPA-CRESTLINE", where the given name is only the first sub-token.
    tokens = [t for t in re.split(r"[\s.\-_/|,]+", name) if t]
    if not tokens:
        return None

    for token in tokens:
        cleaned = token.strip("(){}[]<>\"'`,.;:").strip()
        # A single letter is an initial ("M Varmann", "Anita B"). Skip it
        # and try the next token rather than greeting someone as "Hi M,".
        if len(cleaned) < 2:
            continue
        if cleaned.lower() in _NON_NAME_TOKENS:
            continue
        if not any(ch.isalpha() for ch in cleaned):
            continue
        # WHY .capitalize() rather than leaving as-is: senders write names in
        # caps ("DEEPA"), and shouting the recipient's own name back at
        # them is its own kind of tell.
        return cleaned.capitalize() if cleaned.isupper() else cleaned
    return None


def _looks_like_org(raw: str | None) -> bool:
    if not raw:
        return False
    words = {w.strip(".,").lower() for w in raw.split()}
    return bool(words & _ORG_MARKERS)


def _is_the_company(raw: str | None, opp: Opportunity | None) -> bool:
    """True if this display name is just the employer's name.

    WHY this check exists on top of _ORG_MARKERS: plenty of companies are a
    single ordinary-looking word with no giveaway suffix — "Northwind",
    "Crestline Ites", "STELLARCORP". A marker list cannot catch those and a longer
    marker list would start rejecting real surnames.
    The extractor already identified the employer, so instead of guessing from
    the shape of the string we compare it to a fact we hold: if the sender's
    display name overlaps the company we extracted, it is the company.
    GOTCHA: token overlap rather than equality, because the two rarely match
    exactly — "Northwind" against "Northwind Software", or "Crestline Ites" against
    "Crestline ITES Pvt Ltd".
    """
    if not raw or opp is None or not opp.company:
        return False
    def _tokens(s: str) -> set[str]:
        return {
            t for t in re.split(r"[\s.\-_/|,]+", s.lower())
            if len(t) > 2 and t not in _ORG_MARKERS
        }
    return bool(_tokens(raw) & _tokens(opp.company))


def _salutation(parsed: ParsedMessage, opp: Opportunity | None = None) -> str:
    """Greet the recruiter by name where we can be reasonably sure of it.

    TRACE: two candidate sources, tried in order.
      1. opportunity.recruiter_name — what the extractor pulled from the body
         or the sender's own sign-off. Measured against 22 real messages this
         is markedly better than the header: "Talent Bridge Hr Consulting"
         yields "Arvind K", and "HR Manager" yields "Dinesh Kumar".
      2. parsed.from_name — the From display name, used only when it does not
         look like an organisation. Without a corroborating extracted name,
         "Midwest Consultants Pune" is a company and gets "there".
    Anything unresolved falls through to "there", which is always safe.
    """
    extracted = opp.recruiter_name if opp else None
    name = _first_name(extracted)
    if name:
        return name

    # Only trust the header when it neither advertises itself as a company nor
    # simply repeats the employer we extracted. An org-looking header WITH an
    # extracted name was already handled above; one WITHOUT has nothing to
    # corroborate it.
    if not _looks_like_org(parsed.from_name) and not _is_the_company(parsed.from_name, opp):
        name = _first_name(parsed.from_name)
        if name:
            return name

    return "there"


def _about_role_clause(opp: Opportunity | None) -> str:
    """Optional " about the <role> role at <company>" fragment.

    Returns "" if we have neither. Prefixed with a space so it slots into
    "Thanks for reaching out{about_role}." naturally.
    """
    if opp is None:
        return ""
    role = opp.role_title
    company = opp.company
    if role and company:
        return f" about the {role} role at {company}"
    if role:
        return f" about the {role} role"
    if company:
        return f" about the role at {company}"
    return ""


def _expected_ctc(expected: float | None) -> str:
    """Render the expectation as a negotiable figure.

    WHY "(negotiable)" is attached here rather than typed into candidate.toml:
    it is a property of how we open a conversation, not a fact about the
    candidate. A number stated flatly invites a yes/no; the same number marked
    negotiable invites a counter-offer, which is the point of a first reply.
    GOTCHA: an unset expectation must stay bare "NA" — "NA (negotiable)" reads
    as nonsense and advertises an unconfigured profile.
    """
    if expected is None:
        return NA
    return f"{render(expected)} LPA (negotiable)"


def _budget_accepted(opp: Opportunity | None, profile: CandidateProfile | None) -> bool:
    """True when the JD states a budget we are willing to work with.

    CONCEPT: why this is only ever True on an interested draft.
      The ctc_floor rule (app/rules) already declines anything whose stated
      ceiling falls below `rules.ctc_floor_lpa`, and a declined message never
      reaches this template. So by the time we are writing an INTERESTED
      reply, a stated ctc_max is necessarily at or above the floor. The
      comparison below is therefore belt-and-braces — it makes the invariant
      explicit at the point it matters instead of leaving the reader to infer
      it from a rule in another module.
    """
    if opp is None or profile is None or opp.ctc_max_lpa is None:
        return False
    return float(opp.ctc_max_lpa) >= float(profile.rules.ctc_floor_lpa)


def _budget_line(opp: Opportunity | None, profile: CandidateProfile | None) -> str:
    """Accept a stated budget outright, rather than negotiating against it.

    WHY accept instead of restating our expectation: once the recruiter has
    named a number that clears the floor, haggling costs a round-trip and
    risks the conversation for a difference we already decided we can live
    with. Saying yes and asking for the interview is the shorter path to the
    thing we actually want.
    """
    if not _budget_accepted(opp, profile):
        return ""
    ceiling = render(float(opp.ctc_max_lpa))
    return (
        f"The budget you've mentioned (up to {ceiling} LPA) works for me, "
        f"so no concerns there.\n\n"
    )


def _clarifications_block(
    opp: Opportunity, profile: CandidateProfile | None
) -> str:
    """The bulleted questions, or nothing when there is nothing to ask."""
    items = _clarifications(opp, skip_budget=_budget_accepted(opp, profile))
    if not items:
        return ""
    return "A few things I'd want to confirm before moving forward:\n" + items + "\n\n"


def _closing(
    opp: Opportunity | None,
    profile: CandidateProfile | None,
    resume_attached: bool = False,
) -> str:
    """The last line, which depends on two independent facts.

    GOTCHA: "Happy to share my CV" next to an actual attachment reads as
    though nobody looked at the email before it went out — the recruiter can
    see the PDF. Offering something already provided is the single most
    obvious tell of a template, so the wording has to know what the MIME part
    is doing.
    WHY a different closing once the budget is agreed: the natural next step
    has changed. Before, we are still qualifying the role; after, the only
    remaining question is whether the two of us should talk.
    """
    if resume_attached and _budget_accepted(opp, profile):
        return (
            "I've attached my CV — could we set up an interview at a time "
            "that suits you?"
        )
    if resume_attached:
        return "I've attached my CV as requested — happy to discuss further."
    if _budget_accepted(opp, profile):
        return (
            "Happy to share my CV — could we set up an interview at a time "
            "that suits you?"
        )
    return "Happy to share my CV if that would help."


def _clarifications(opp: Opportunity, skip_budget: bool = False) -> str:
    """Bullet list of things missing from the JD that we'd want confirmed.

    Deterministic: same inputs → same bullets. No LLM.
    """
    items: list[str] = []
    if not opp.jd_text:
        items.append("could you share the full JD?")
    if opp.employment_type is None:
        items.append("is this permanent or contract? (I do not consider c2h)")
    elif opp.employment_type == "contract":
        items.append("please confirm the contract duration and extension terms")
    # WHY skip_budget: when the stated ceiling already clears our floor we have
    # just said so a few lines above. Asking "what is the budget?" immediately
    # after accepting their budget reads as though nobody proofread the mail.
    if not skip_budget and opp.ctc_min_lpa is None and opp.ctc_max_lpa is None:
        items.append("what is the budget range for this role?")
    if opp.notice_period is None:
        items.append("what is the expected joining timeline?")
    if opp.work_model is None:
        items.append("is this onsite, hybrid, or remote?")

    if not items:
        # WHY empty rather than a filler bullet: this used to append "interview
        # process and expected timeline" so the "before moving forward:" line
        # was never left dangling. _clarifications_block now omits the heading
        # entirely when there is nothing to ask, which is the better answer —
        # inventing a question purely to justify a heading is padding, and the
        # budget-accepted path already ends by asking for the interview.
        return ""

    return "\n".join(f"  - {it}" for it in items)


# _num() lived here until profile fields became optional. Its float
# formatting now lives in candidate.render(), which also handles None — one
# function so a draft and the fit-scorer prompt cannot format the same value
# two different ways.
