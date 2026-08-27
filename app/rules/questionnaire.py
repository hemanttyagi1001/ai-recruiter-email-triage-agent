"""
Is this follow-up a screening form, and which fields does it ask for?

What this module does and why it exists here: D55 removed
`followup_existing_thread` from the extractable categories because the agent
sees exactly one message and has no thread history, so anything it wrote into
an ongoing conversation read as a non-sequitur. That argument is correct for a
CTC negotiation, a rejection, or a closure. It is NOT correct for a screening
questionnaire, because the answers to one are standing facts about the
candidate — identical no matter what was said earlier in the thread. This
module finds that narrow, safe subset and nothing else.

CONCEPT: the seam is "does answering require history?", not "is it a follow-up".
  D55's real constraint was never the category label. It was that the agent
  cannot answer a question whose meaning depends on context it cannot see. A
  form asking "Current CTC:" depends on no context at all. So the test applied
  here is not "is this a follow-up" but "does this message ask only for things
  the profile already knows".

WHY code and not a prompt, and not a classifier label: it decides whether an
  email is sent, so D11 applies — the same reason the decline rules and the
  reply-target resolver are regex and comparisons. There is a sharper reason
  too. The trigger text is written by a stranger, so a model asked "is this a
  questionnaire?" can be talked into yes by an email that says it is one.
  Counting known field labels cannot be argued with.

GOTCHA: the threshold is what keeps this from reopening D55 by accident.
  One mention of "CTC" is a negotiation — the single most common follow-up
  there is, and exactly the case D55 was written about. A colon-terminated
  list of six labels is a form. `MIN_FIELDS = 3` is deliberately set above the
  point where the two are distinguishable; missing a real form costs a manual
  reply, while answering a negotiation with a wall of salary figures is a
  message that cannot be taken back.
"""

from __future__ import annotations

import re

# CONCEPT: each entry is (canonical label, pattern). The canonical label is
# what gets echoed back in the reply, so a recruiter's "Total Exp:" comes home
# as "Total Experience" — tidy without inventing anything, since the answer is
# the same either way.
# GOTCHA: patterns require a trailing colon (optionally after whitespace).
# That single character is most of the discrimination here: prose that happens
# to mention "your current CTC" does not match, while a form field does. The
# observed forms are all colon-terminated because they are meant to be filled
# in underneath.
_FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Current Location", r"current\s+location"),
    ("Native Location", r"native\s+location"),
    ("Total Experience", r"total\s+(?:experience|exp)"),
    ("Relevant Experience", r"relevant\s+(?:experience|exp)"),
    ("Preferred Location", r"preferred\s+location(?:\s+to\s+work)?"),
    ("Reason for Job Change", r"reason\s+for\s+(?:job\s+)?change"),
    ("Notice Period", r"notice\s+period"),
    ("Current CTC", r"current\s+ctc"),
    ("Expected CTC", r"expected\s+ctc"),
    ("Last Working Day", r"last\s+working\s+day"),
)

# GOTCHA: the label and its colon are frequently separated by a parenthetical
# instruction. The message that motivated this module wrote:
#     Notice Period (If currently not working, please mention last working day):
# A pattern demanding `notice\s+period\s*:` misses that line entirely, which
# silently drops the one field a recruiter most wants answered. Allowing an
# optional bracketed aside between label and colon is what makes the real form
# parse — the first version of this was tuned on invented samples and got it
# wrong on the only real one.
# WHY the aside is not itself scanned for labels: "last working day" appears
# inside that parenthetical as part of the notice-period question, not as a
# field of its own. Matching it separately would answer the same question
# twice.
_COLON = r"(?:\s*\([^)]*\))?\s*:"

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern + _COLON, re.IGNORECASE))
    for label, pattern in _FIELD_PATTERNS
)

# Below this many distinct fields, treat the message as ordinary follow-up
# conversation and leave it to a human. See the module GOTCHA.
MIN_FIELDS = 3


def detect_fields(body_text: str | None) -> list[str]:
    """Canonical labels this message asks for, in the order it asks for them.

    WHY order matters enough to preserve: a recruiter pastes the reply into an
    ATS field by field. Answering in their order makes that a copy job;
    answering in ours makes them hunt. Sorting by first match position is the
    whole cost of getting that right.

    TRACE: pure function over the already-scrubbed body. No I/O, no LLM, same
    answer every time for the same email — which is the property that lets the
    routing decision built on it be reviewed rather than trusted.
    """
    if not body_text:
        return []
    found: list[tuple[int, str]] = []
    for label, pattern in _COMPILED:
        m = pattern.search(body_text)
        if m is not None:
            found.append((m.start(), label))
    found.sort()
    return [label for _, label in found]


def is_profile_questionnaire(body_text: str | None) -> bool:
    """True if this message is a screening form asking for standing facts.

    GOTCHA: this is necessary but NOT sufficient to answer automatically. The
    caller must ALSO have classified the message as
    `followup_existing_thread`. A new role pitch that happens to list three
    field labels is still a pitch and belongs on the normal path, where it
    gets extracted, scored and answered on its merits. Category and content
    both have to agree before D55 is set aside for a message.
    """
    return len(detect_fields(body_text)) >= MIN_FIELDS
