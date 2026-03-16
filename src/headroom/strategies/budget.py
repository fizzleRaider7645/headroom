from __future__ import annotations

from typing import Literal

from headroom.core.budget import TokenBudget
from headroom.core.message import TrackedMessage
from headroom.strategies.base import BaseStrategy, SessionContext


class BudgetGuardStrategy(BaseStrategy):
    """
    Last-resort strategy: drops oldest non-pinned messages when in overflow.
    Fires only when the context is over the hard limit (other strategies failed).
    """

    priority = 10

    def __init__(
        self,
        overflow_action: Literal["drop_oldest", "raise"] = "drop_oldest",
        drop_batch_size: int = 2,
    ):
        self.overflow_action = overflow_action
        self.drop_batch_size = drop_batch_size

    def should_apply(self, budget: TokenBudget, used_tokens: int) -> bool:
        return self.enabled and budget.status(used_tokens) == "overflow"

    @property
    def params(self) -> list[dict]:
        return [
            {
                "name": "drop_batch_size",
                "label": "Drop batch size",
                "type": "int",
                "value": self.drop_batch_size,
                "min": 1,
                "max": 10,
            },
        ]

    def apply(
        self,
        messages: list[TrackedMessage],
        budget: TokenBudget,
        used_tokens: int,
        ctx: SessionContext,
    ) -> list[TrackedMessage]:
        if self.overflow_action == "raise":
            raise ContextOverflowError(
                f"Context overflow: {used_tokens} tokens used, limit is {budget.limit}"
            )

        result = list(messages)
        # Estimate token reduction per message drop (heuristic)
        while budget.status(used_tokens) == "overflow" and result:
            # Find oldest non-pinned messages
            droppable = [i for i, m in enumerate(result) if not m.pinned]
            if not droppable:
                break  # everything is pinned, can't drop anything
            drop_indices = set(droppable[: self.drop_batch_size])
            dropped_tokens = sum(
                result[i].token_count or 100 for i in drop_indices
            )
            result = [m for i, m in enumerate(result) if i not in drop_indices]
            used_tokens = max(0, used_tokens - dropped_tokens)

        return result


class ContextOverflowError(Exception):
    pass
