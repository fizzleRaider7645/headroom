from unittest.mock import MagicMock

from headroom.core.budget import TokenBudget
from headroom.core.message import TrackedMessage
from headroom.strategies.base import SessionContext
from headroom.strategies.cache import CacheInjectionStrategy


def _ctx():
    return SessionContext(client=MagicMock(), model="claude-opus-4-6")


def _make_msg(role="user", content="hello world", pinned=False, tokens=0):
    m = TrackedMessage(role=role, content=content, pinned=pinned)
    m.token_count = tokens
    return m


def test_cache_control_injected_on_pinned():
    strategy = CacheInjectionStrategy()
    budget = TokenBudget(limit=10_000)
    pinned = _make_msg(content="Big system context", pinned=True, tokens=2000)
    other = _make_msg(content="Normal message")
    messages = [pinned, other]

    result = strategy.apply(messages, budget, 500, _ctx())

    # pinned message should have cache breakpoint
    pinned_result = next(m for m in result if m.pinned)
    assert pinned_result.cache_breakpoint is True
    # Content should now be a list with cache_control
    assert isinstance(pinned_result.content, list)
    assert "cache_control" in pinned_result.content[0]


def test_string_content_converted_to_list_on_cache():
    strategy = CacheInjectionStrategy()
    budget = TokenBudget(limit=10_000)
    pinned = _make_msg(content="Context data here", pinned=True)
    messages = [pinned, _make_msg(), _make_msg(), _make_msg()]

    result = strategy.apply(messages, budget, 100, _ctx())
    pinned_result = result[0]

    if pinned_result.cache_breakpoint:
        assert isinstance(pinned_result.content, list)
        assert pinned_result.content[0]["type"] == "text"
        assert "cache_control" in pinned_result.content[0]


def test_max_breakpoints_not_exceeded():
    strategy = CacheInjectionStrategy(max_breakpoints=2)
    budget = TokenBudget(limit=10_000)
    messages = [
        _make_msg(content=f"Message {i}", pinned=(i == 0), tokens=1500)
        for i in range(8)
    ]
    result = strategy.apply(messages, budget, 500, _ctx())
    breakpoints = sum(1 for m in result if m.cache_breakpoint)
    assert breakpoints <= 2


def test_always_applies():
    strategy = CacheInjectionStrategy()
    budget = TokenBudget(limit=10_000)
    # Even when budget is "ok", cache strategy still runs
    assert strategy.should_apply(budget, 100) is True
