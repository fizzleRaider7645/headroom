from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from headroom.core.budget import TokenBudget
    from headroom.core.message import TrackedMessage


@dataclass
class SessionContext:
    """Injectable context for strategies that need API access."""

    client: anthropic.Anthropic
    model: str
    system: str | None = None


class BaseStrategy(ABC):
    """
    Base class for all context optimization strategies.

    Each strategy receives the full message list and returns a (possibly shorter)
    modified copy. Strategies must not mutate the input list.
    """

    priority: int = 50  # lower = runs earlier in the pipeline
    enabled: bool = True  # can be toggled at runtime without removing from pipeline

    @abstractmethod
    def apply(
        self,
        messages: list["TrackedMessage"],
        budget: "TokenBudget",
        used_tokens: int,
        ctx: SessionContext,
    ) -> list["TrackedMessage"]:
        """Return a new message list, potentially shorter or modified."""

    def should_apply(self, budget: "TokenBudget", used_tokens: int) -> bool:
        """Default: fire when budget status is 'act' or 'overflow'.

        Always returns False when the strategy is disabled.
        """
        if not self.enabled:
            return False
        return budget.status(used_tokens) in ("act", "overflow")

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def params(self) -> list[dict]:
        """Configurable parameters for this strategy.

        Each entry is a dict with keys:
          name   – attribute name on the strategy instance
          label  – human-readable label for the UI
          type   – "int" or "float"
          value  – current value
          min    – minimum allowed value
          max    – maximum allowed value
          step   – input step (optional, defaults to 1 for int / 0.05 for float)

        Override in subclasses to expose parameters.
        """
        return []

    def set_param(self, name: str, raw_value: str) -> None:
        """Update a named parameter, casting from the string value the UI sends."""
        for p in self.params:
            if p["name"] == name:
                if p["type"] == "int":
                    casted = int(float(raw_value))
                    casted = max(p["min"], min(p["max"], casted))
                elif p["type"] == "float":
                    casted = round(float(raw_value), 6)
                    casted = max(p["min"], min(p["max"], casted))
                else:
                    raise ValueError(f"Unknown param type '{p['type']}'")
                setattr(self, name, casted)
                return
        raise ValueError(f"Unknown param '{name}' for strategy '{self.name}'")
