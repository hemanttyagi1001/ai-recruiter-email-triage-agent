"""
The one library warning we suppress, and everything we must NOT suppress.

Why this file is mostly negative assertions: a warning filter is a promise
that you understood the warning and that it cannot become a real signal. The
only way that promise stays true is if the filter is narrow, and the only way
to know it stayed narrow is to assert what still gets through. A filter that
silently widened would look identical in production right up to the moment it
hid something worth seeing.

Background: langsmith's wrap_openai dumps the ParsedChatCompletion returned by
beta.chat.completions.parse to build its trace payload. That response's
`parsed` field is generic; at dump time it resolves to NoneType, so Pydantic
warns it found a model where None was declared. Cosmetic, but it fires once
per structured LLM call — three per message — and drowns the log.
"""

from __future__ import annotations

import warnings

import pytest

from app.observability import tracing


@pytest.fixture(autouse=True)
def _fresh_filters():
    """Each test gets a clean warning-filter stack.

    GOTCHA: warnings.filterwarnings mutates PROCESS-GLOBAL state, and the
    module guards itself with a `_warning_filtered` flag. Both have to be
    reset, or the second test in this file silently exercises the first
    test's filter.
    """
    tracing._warning_filtered = False
    with warnings.catch_warnings():
        warnings.resetwarnings()
        yield
    tracing._warning_filtered = False


# The real message, copied verbatim from a traced run.
REAL_WARNING = (
    "Pydantic serializer warnings:\n"
    "  PydanticSerializationUnexpectedValue(Expected `none` - serialized "
    "value may not be as expected [field_name='parsed', "
    "input_value=ClassificationResult(reason='...', category='new_role_pitch',"
    " confidence=0.95), input_type=ClassificationResult])"
)


def _emitted(message: str, category=UserWarning) -> bool:
    """True if `message` survives the currently installed filters."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn(message, category)
        return len(caught) > 0


def test_the_langsmith_artefact_is_silenced():
    tracing.silence_parsed_completion_warning()
    assert not _emitted(REAL_WARNING)


def test_it_is_noisy_again_without_the_filter():
    """Proves the test above is testing the filter and not the harness."""
    assert _emitted(REAL_WARNING)


# --- what must still get through --------------------------------------------


def test_other_pydantic_serializer_warnings_still_surface():
    """A serializer warning about a DIFFERENT field is a real signal.

    If our own models ever start round-tripping wrongly, this is how we would
    hear about it — so the filter must not key on the "Pydantic serializer
    warnings:" prefix alone.
    """
    tracing.silence_parsed_completion_warning()
    other = (
        "Pydantic serializer warnings:\n"
        "  PydanticSerializationUnexpectedValue(Expected `str` - serialized "
        "value may not be as expected [field_name='draft_body', "
        "input_value=None, input_type=NoneType])"
    )
    assert _emitted(other)


def test_unrelated_user_warnings_still_surface():
    tracing.silence_parsed_completion_warning()
    assert _emitted("token.json is missing the gmail.modify scope")


def test_deprecation_warnings_are_untouched():
    tracing.silence_parsed_completion_warning()
    assert _emitted("datetime.utcnow() is deprecated", DeprecationWarning)


# --- installation discipline ------------------------------------------------


def test_calling_twice_installs_one_filter():
    """Idempotent: wrap_llm_client may run once per LLMClient, and a filter
    list that grows on every construction is its own slow leak."""
    before = len(warnings.filters)
    tracing.silence_parsed_completion_warning()
    after_first = len(warnings.filters)
    tracing.silence_parsed_completion_warning()
    tracing.silence_parsed_completion_warning()
    assert after_first == before + 1
    assert len(warnings.filters) == after_first


def test_an_untraced_process_installs_no_filter(monkeypatch):
    """No wrap_openai, no filter.

    The filter exists to hide an artefact of tracing. A process with tracing
    off has no artefact to hide, and should not have its warning behaviour
    quietly altered by importing this module.
    """
    monkeypatch.setattr(tracing, "get_client", lambda: None)
    sentinel = object()
    before = len(warnings.filters)

    returned = tracing.wrap_llm_client(sentinel)

    assert returned is sentinel          # identity function when untraced
    assert len(warnings.filters) == before
    assert tracing._warning_filtered is False
