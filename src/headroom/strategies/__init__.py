from headroom.strategies.base import BaseStrategy, SessionContext
from headroom.strategies.budget import BudgetGuardStrategy, ContextOverflowError
from headroom.strategies.cache import CacheInjectionStrategy
from headroom.strategies.relevance import RelevanceFilterStrategy
from headroom.strategies.summarizer import SummarizationStrategy

__all__ = [
    "BaseStrategy",
    "SessionContext",
    "BudgetGuardStrategy",
    "ContextOverflowError",
    "CacheInjectionStrategy",
    "RelevanceFilterStrategy",
    "SummarizationStrategy",
]


def default_strategies() -> list[BaseStrategy]:
    """Return the default strategy pipeline (all strategies, sensible config)."""
    return [
        BudgetGuardStrategy(),
        RelevanceFilterStrategy(),
        SummarizationStrategy(),
        CacheInjectionStrategy(),
    ]
