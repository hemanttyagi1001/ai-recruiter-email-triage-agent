"""
Shared types for the rules package. Split out from engine.py to avoid a
circular import: engine.py imports the individual rule callables from
decliners.py, and decliners.py needs the RuleVerdict type. Keeping the
type here lets both sides import from a leaf module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.schemas import Opportunity


Verdict = Literal["decline"]


@dataclass(frozen=True)
class RuleVerdict:
    rule_name: str
    verdict: Verdict
    reason: str


Rule = Callable[["Opportunity"], "RuleVerdict | None"]
