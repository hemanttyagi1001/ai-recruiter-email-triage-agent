"""
Thin wrapper around LLMClient.embed() for jd_text embedding.

Why this module exists instead of calling llm.embed() directly from the
node: it gives us one place to normalise/prepare the text before it hits
the embedding API. Today that's just "strip whitespace"; tomorrow it might
include collapsing recruiter-template boilerplate, capping length below
the model's token limit, or stripping quoted-reply chains that leak into
jd_text from a followup.

CONCEPT: what an embedding is, in one paragraph.
  An embedding is a fixed-length dense vector — 1536 floats for
  text-embedding-3-small — produced by a model trained so that texts
  with similar meaning land in nearby regions of that 1536-dim space.
  Not a token count, not a TF-IDF sparse vector, not a hash — a learned
  compression of meaning. Two paraphrased JDs ("senior .NET developer"
  and "experienced C# engineer") land close together in that space; two
  unrelated JDs land far apart. This is what lets dedup survive
  paraphrase — we're comparing shapes-of-meaning, not shapes-of-strings.

CONCEPT: why we prepare text before embedding.
  Embedding models charge per input token and have hard token limits
  (text-embedding-3-small caps at 8191 tokens ≈ 30k chars). More
  importantly, the vector produced summarises whatever we pass in — if
  we pass 8k tokens of recruiter-signature boilerplate, we get a vector
  that partly encodes "recruiter signature" and only partly encodes
  "this specific role." Preparation is where we bias the vector toward
  the signal we care about. Kept minimal in Phase 4; the extractor
  already gave us jd_text stripped of greeting/sign-off.
"""

from __future__ import annotations

from app.llm.client import LLMClient, Usage

# WHY this ceiling: text-embedding-3-small tokenises at roughly 1 token
# per 3-4 chars for English recruiter prose. 20_000 chars stays well
# under the 8191-token API limit while covering essentially every real
# recruiter JD we've seen — the median is closer to 1_500 chars.
# ALTERNATIVE: tokenise with tiktoken and cap at ~7500 tokens. Rejected
# for now — an extra dep and process cost for a ceiling that is already
# 10× the real distribution's tail. Revisit if we start ingesting mail
# with attached JDs pasted inline.
MAX_EMBED_CHARS = 20_000


def prepare_text(jd_text: str) -> str:
    """Normalise jd_text prior to embedding.

    Today: strip surrounding whitespace and cap at MAX_EMBED_CHARS. Kept
    as its own function so tests can assert stable preparation without
    hitting the API, and so future normalisation (whitespace collapse,
    quoted-reply strip) has an obvious home.
    """
    text = jd_text.strip()
    if len(text) > MAX_EMBED_CHARS:
        text = text[:MAX_EMBED_CHARS]
    return text


def embed_jd_text(jd_text: str, llm: LLMClient) -> tuple[list[float], Usage]:
    """Embed a JD string. Returns (vector, usage).

    Callers are responsible for the min-length gate — this function
    embeds whatever it's handed. That split keeps the "should we embed?"
    policy in the node (where it can consult candidate config) and the
    "how do we embed?" mechanics here (where it can be unit-tested
    without a config).
    """
    prepared = prepare_text(jd_text)
    return llm.embed(prepared)
