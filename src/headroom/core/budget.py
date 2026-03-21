from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Default context limits per model
MODEL_LIMITS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
}

BudgetStatus = Literal["ok", "warn", "act", "overflow"]


@dataclass
class TokenBudget:
    """Defines token limits and thresholds for a session."""

    limit: int
    warn_at: float = 0.80
    act_at: float = 0.90
    reserve: int = 1_024
    cost_limit: float | None = None  # optional dollar budget (USD)

    @property
    def usable(self) -> int:
        """Tokens available for conversation (limit minus response reserve)."""
        return self.limit - self.reserve

    def headroom(self, used: int) -> int:
        """Remaining tokens before hitting the usable limit."""
        return max(0, self.usable - used)

    def headroom_fraction(self, used: int) -> float:
        """Fraction of usable context still free (0.0 = full, 1.0 = empty)."""
        if self.usable <= 0:
            return 0.0
        return self.headroom(used) / self.usable

    def used_fraction(self, used: int) -> float:
        """Fraction of usable context consumed (0.0 = empty, 1.0 = full)."""
        if self.usable <= 0:
            return 1.0
        return min(1.0, used / self.usable)

    def status(self, used: int) -> BudgetStatus:
        f = self.used_fraction(used)
        if f >= 1.0:
            return "overflow"
        if f >= self.act_at:
            return "act"
        if f >= self.warn_at:
            return "warn"
        return "ok"

    @classmethod
    def for_model(cls, model: str, **kwargs) -> "TokenBudget":
        limit = MODEL_LIMITS.get(model, 200_000)
        return cls(limit=limit, **kwargs)


@dataclass
class TokenUsage:
    """Snapshot of token usage at a point in time."""

    used: int
    limit: int
    reserve: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_hits: int = 0
    turns: int = 0
    total_cost: float = 0.0        # cumulative USD cost for the session
    cost_limit: float | None = None  # optional dollar budget

    @property
    def headroom(self) -> int:
        return max(0, self.limit - self.reserve - self.used)

    @property
    def used_fraction(self) -> float:
        usable = self.limit - self.reserve
        if usable <= 0:
            return 1.0
        return min(1.0, self.used / usable)

    @property
    def headroom_pct(self) -> int:
        return round((1.0 - self.used_fraction) * 100)

    @property
    def cost_remaining(self) -> float | None:
        """Remaining dollar budget, or None if no cost_limit is set."""
        if self.cost_limit is None:
            return None
        return max(0.0, self.cost_limit - self.total_cost)
