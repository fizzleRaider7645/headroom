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
        """Default: fire when budget status is 'act' or 'overflow'."""
        return budget.status(used_tokens) in ("act", "overflow")

    @property
    def name(self) -> str:
        return self.__class__.__name__
