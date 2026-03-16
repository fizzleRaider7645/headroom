from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from headroom.strategies.base import BaseStrategy, SessionContext

if TYPE_CHECKING:
    from headroom.core.budget import TokenBudget
    from headroom.core.message import TrackedMessage

# Anthropic's maximum cache breakpoints per request
MAX_BREAKPOINTS = 4

# Minimum tokens before a cache breakpoint is worth placing
# (models have different minimums; 1024 is conservative)
MIN_CACHE_TOKENS = 1_024


class CacheInjectionStrategy(BaseStrategy):
    """
    Injects cache_control breakpoints into the message list to reduce API costs.

    Identifies the stable prefix (messages sent in the previous turn unchanged)
    and places breakpoints at logical boundaries.

    Always runs last (priority=90) after other strategies have trimmed the list.
    """

    priority = 90

    def __init__(
        self,
        max_breakpoints: int = MAX_BREAKPOINTS,
        min_cache_tokens: int = MIN_CACHE_TOKENS,
    ):
        self.max_breakpoints = min(max_breakpoints, MAX_BREAKPOINTS)
        self.min_cache_tokens = min_cache_tokens

    def should_apply(self, budget: "TokenBudget", used_tokens: int) -> bool:
        return self.enabled  # always runs unless explicitly disabled

    @property
    def params(self) -> list[dict]:
        return [
            {
                "name": "max_breakpoints",
                "label": "Max breakpoints",
                "type": "int",
                "value": self.max_breakpoints,
                "min": 1,
                "max": MAX_BREAKPOINTS,
            },
            {
                "name": "min_cache_tokens",
                "label": "Min cache tokens",
                "type": "int",
                "value": self.min_cache_tokens,
                "min": 256,
                "max": 4096,
            },
        ]

    def apply(
        self,
        messages: list["TrackedMessage"],
        budget: "TokenBudget",
        used_tokens: int,
        ctx: SessionContext,
    ) -> list["TrackedMessage"]:
        # Work on deep copies so we don't mutate the originals
        result = [copy.copy(m) for m in messages]

        # Find candidate breakpoint positions
        candidates: list[int] = []

        # 1. End of pinned/summary messages block
        last_pinned = -1
        for i, m in enumerate(result):
            if m.pinned:
                last_pinned = i
        if last_pinned >= 0:
            candidates.append(last_pinned)

        # 2. System / long context injections (large messages near the start)
        for i, m in enumerate(result[: len(result) // 2]):
            tokens = m.token_count or 0
            if tokens >= self.min_cache_tokens and i not in candidates:
                candidates.append(i)

        # 3. End of stable prefix (second-to-last message if list is long)
        if len(result) >= 4 and (len(result) - 2) not in candidates:
            candidates.append(len(result) - 2)

        # Limit to max_breakpoints, prefer later positions (more stable cache)
        candidates = sorted(set(candidates))[-self.max_breakpoints :]

        # Inject cache_control into selected messages
        for idx in candidates:
            msg = result[idx]
            msg.cache_breakpoint = True
            if isinstance(msg.content, str):
                msg.content = [
                    {
                        "type": "text",
                        "text": msg.content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(msg.content, list):
                # Inject into the last text block that doesn't already have cache_control
                for block in reversed(msg.content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        if "cache_control" not in block:
                            block["cache_control"] = {"type": "ephemeral"}
                        break

        return result
