"""
Did the recruiter actually ask for a CV?

What this module does and why it exists here: attaching a resume is a small
outbound decision with a real cost when it is wrong. Attach unasked and the
reply looks presumptuous; fail to attach when asked and the recruiter has to
chase, which is the exact friction this agent exists to remove. This answers
that one question from the message body, deterministically.

WHY code and not a prompt, and not an extractor field:
  The standing rule (D11) is that a decision which gates an outbound action
  lives in code. But there is a sharper reason here. The phrase that triggers
  an attachment appears in text written by a stranger — so an extractor field
  like `resume_requested: bool` would let a crafted email talk the model into
  "yes". A regex over a fixed pattern list cannot be argued with, and a reader
  can audit exactly which sentences fire it.

CONCEPT: the discriminator is grammatical mood, not the word "resume".
  Measured against 118 real messages, the word appears constantly in contexts
  that are NOT requests:
    - "I submitted your resume to Elektrobit"        (already done, by them)
    - "thank you for sharing your resume with us"    (already received)
    - "block this recruiter from searching your resume"
    - "NLP-based resume parsing and candidate matching"  (a JD skill!)
    - "Increase your chances by keeping your resume up-to-date"  (an ad)
  Every genuine request is an imperative aimed at the reader:
    - "kindly share your updated resume to Arvind@talentbridge.com"
    - "Kindly revert with your updated resume along with below details"
    - "Interested candidates can share their resumes to ..."
    - "please send your updated word resume inline to the JD"
GOTCHA: the verbs below are matched with a trailing \\b on the BASE form, which
is what excludes the past and gerund traps for free — "share" does not match
"sharing", and "submit" does not match "submitted". That is load-bearing, not
incidental; dropping the boundary would fire on half the false positives above.
"""

from __future__ import annotations

import re

# Base-form request verbs only. See the module GOTCHA — inflected forms are
# almost always someone describing what already happened, not asking.
_REQUEST_VERBS = (
    r"share|send|forward|attach|provide|submit|mail|e-?mail|revert\s+with|"
    r"reply\s+with|respond\s+with|drop|ping\s+me"
)

# The artefact being asked for. "profile" is included because Indian recruiting
# uses it interchangeably with CV; the cost of a false positive here is one
# unnecessary attachment, against a false negative costing a round-trip.
_ARTEFACT = r"(?:updated\s+|latest\s+|word\s+|current\s+)*(?:resumes?|cvs?|profiles?|candidature)"

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "kindly share your updated resume to ..." — the dominant form. The gap
    # allows for "your", "their", "me the", "us your" without enumerating them.
    re.compile(
        rf"(?i)\b(?:{_REQUEST_VERBS})\b[^.\n]{{0,40}}\b{_ARTEFACT}\b"
    ),
    # "Apply with resume & profile" — no explicit send verb.
    re.compile(r"(?i)\bapply\s+with\b[^.\n]{0,20}\b(?:resumes?|cvs?)\b"),
    # "awaiting your resume" / "looking forward to your CV"
    re.compile(r"(?i)\b(?:awaiting|expecting)\b[^.\n]{0,25}\b(?:resumes?|cvs?)\b"),
)


def is_resume_requested(body_text: str | None) -> bool:
    """True if the message asks the recipient to send a CV.

    TRACE: runs once per message straight after extraction, on the same
    already-scrubbed body the classifier saw. Pure function, no I/O, no LLM —
    so it costs nothing and returns the same answer for the same email every
    time, which matters because the alternative (asking the model) would not.
    """
    if not body_text:
        return False
    return any(p.search(body_text) for p in _PATTERNS)


def _is_naukri_sender(from_email: str | None) -> bool:
    """True if this arrived through Naukri's relay.

    WHY the check is on the FROM header rather than on `source_platform` from
    the extractor: the header is stamped by the relay itself and cannot be
    influenced by the message body, while an extracted field is the model's
    reading of text a stranger wrote. This gates an attachment, so it follows
    the same rule as everything else on the outbound path — derive it from
    something the sender cannot author. See D11.
    """
    if not from_email or "@" not in from_email:
        return False
    domain = from_email.strip().lower().rpartition("@")[2]
    return domain == "naukri.com" or domain.endswith(".naukri.com")


def should_attach_resume(body_text: str | None, from_email: str | None) -> bool:
    """True if this reply should carry the CV.

    CONCEPT: two independent reasons to attach, one flag.
      1. The recruiter asked — `is_resume_requested` above, matching an
         imperative in the body.
      2. The message came through Naukri. Relay mail rarely contains an
         explicit request because the portal's own flow is "apply and we
         forward your profile", so the ask never appears as a sentence. The
         recruiter on the other end still expects a CV, and by the time they
         have to ask for one the thread has cost a round-trip.

    GOTCHA: the caller stores this on `resume_requested`, whose name is now
    narrower than its meaning — it answers "attach?", not "was it asked for?".
    Renaming the state key would touch the graph, both send nodes and their
    tests for no behavioural gain, so the name stays and this note exists
    instead. What must NOT drift is the pairing: the same flag decides the
    closing sentence and the MIME part, so they cannot contradict each other
    (D61).
    """
    return is_resume_requested(body_text) or _is_naukri_sender(from_email)
