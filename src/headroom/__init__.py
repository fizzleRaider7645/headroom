"""
headroom — optimize context and token usage for Claude API conversations.

Quick start::

    from headroom import Session, TokenBudget

    session = Session(model="claude-opus-4-6", budget=TokenBudget(limit=200_000))
    response = session.send_sync("Hello!")
    print(f"Headroom: {session.token_usage.headroom} tokens remaining")
"""

from headroom.core.budget import TokenBudget, TokenUsage
from headroom.core.message import TrackedMessage
from headroom.core.session import Session, BudgetEvent, TrimEvent
from headroom.strategies import (
    BaseStrategy,
    BudgetGuardStrategy,
    CacheInjectionStrategy,
    RelevanceFilterStrategy,
    SummarizationStrategy,
    default_strategies,
)

__version__ = "0.1.0"

__all__ = [
    "Session",
    "TokenBudget",
    "TokenUsage",
    "TrackedMessage",
    "BudgetEvent",
    "TrimEvent",
    "BaseStrategy",
    "BudgetGuardStrategy",
    "CacheInjectionStrategy",
    "RelevanceFilterStrategy",
    "SummarizationStrategy",
    "default_strategies",
]
