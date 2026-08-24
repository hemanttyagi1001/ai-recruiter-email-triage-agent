"""
Red-team check on LLMClient.embed().

Phase 4 added a second Azure OpenAI surface (embeddings) alongside the
existing chat-completions surface. The trust-boundary claim from D13 —
"an LLM call cannot fire an HTTP request, spawn a shell, delete a row,
or write a file, because the surface exposed to the LLM is `str/int/enum
→ BaseModel`" — has to extend to the new surface, or D13 is silently
weakened.

This file verifies the structural claim by inspection. If someone adds
a `tools=` parameter, a function-calling method, or turns the embed
call into something that accepts callable arguments, this test breaks
— which is the intended alarm.

Runtime behaviour tests for the embed method live under tests/ proper;
this file is only about the safety-relevant *surface*.
"""

from __future__ import annotations

import inspect

from app.llm.client import LLMClient


def test_embed_signature_has_no_tools_parameter():
    sig = inspect.signature(LLMClient.embed)
    assert "tools" not in sig.parameters, (
        "LLMClient.embed grew a `tools` parameter. Embeddings must not have "
        "a tool surface — the D13 trust boundary applies to every LLM "
        "surface in this codebase, including embeddings (see D24)."
    )
    # The only positional/keyword params should be `self` and `text`.
    # `text` must be a plain str, not a callable or a schema.
    non_self = [
        p for name, p in sig.parameters.items() if name != "self"
    ]
    assert len(non_self) == 1, (
        f"LLMClient.embed signature changed: {sig}. Any new parameter is a "
        f"new attack surface — add a dedicated check for it here."
    )
    text_param = non_self[0]
    assert text_param.name == "text"
    # Best-effort annotation check (str). If we ever add a broader type,
    # this fails so we notice.
    # GOTCHA: client.py uses `from __future__ import annotations`, so
    # inspect returns annotations as strings, not classes. Accept both.
    assert text_param.annotation in (str, "str", inspect.Parameter.empty), (
        f"LLMClient.embed's text parameter is no longer str-only "
        f"(got {text_param.annotation!r}). If we now accept richer input, "
        f"prove it can't be a callable / schema / tool-like object."
    )


def test_embed_return_annotation_is_vector_and_usage():
    """The advertised return type is (list[float], Usage). Nothing else.
    A return type that included a callable or a "run this" object would
    be an escape from the trust boundary."""
    sig = inspect.signature(LLMClient.embed)
    ann = sig.return_annotation
    # We don't need to unpack the full Generic; just make sure it's still
    # a tuple-shaped return. If someone changes it to return a live client
    # or an executable object, this fires.
    assert ann is not inspect.Signature.empty, (
        "LLMClient.embed lost its return annotation — please restore it "
        "so this red-team check can compare it."
    )
    ann_str = str(ann)
    assert "list[float]" in ann_str and "Usage" in ann_str, (
        f"LLMClient.embed's return signature changed to {ann_str!r}. "
        f"The declared return must remain (list[float], Usage) — any richer "
        f"return type needs a fresh trust-boundary review."
    )
