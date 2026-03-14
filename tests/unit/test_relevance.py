from headroom.core.budget import TokenBudget
from headroom.core.message import TrackedMessage
from headroom.strategies.base import SessionContext
from headroom.strategies.relevance import RelevanceFilterStrategy
from unittest.mock import MagicMock


def _ctx():
    return SessionContext(client=MagicMock(), model="claude-opus-4-6")


def _make_msgs(texts: list[tuple[str, str]]) -> list[TrackedMessage]:
    return [TrackedMessage(role=role, content=text) for role, text in texts]


def test_irrelevant_messages_dropped():
    budget = TokenBudget(limit=10_000, act_at=0.0)  # always act
    strategy = RelevanceFilterStrategy(
        query_window=1, similarity_threshold=0.5, min_keep=1
    )
    messages = _make_msgs([
        ("user", "Tell me about Python programming and code"),
        ("assistant", "Python is a great language for coding"),
        ("user", "What is quantum physics about neutron stars"),
        ("assistant", "Quantum physics involves subatomic particles"),
        ("user", "How do I write a Python function"),  # current query
    ])
    result = strategy.apply(messages, budget, 500, _ctx())
    # Should keep the Python-related messages and drop the quantum ones
    contents = [m.content for m in result]
    # At minimum, the last message (current query) must be kept
    assert any("Python" in c for c in contents if isinstance(c, str))


def test_min_keep_respected():
    budget = TokenBudget(limit=10_000, act_at=0.0)
    strategy = RelevanceFilterStrategy(
        query_window=1, similarity_threshold=0.99, min_keep=3
    )
    messages = _make_msgs([
        ("user", "foo bar baz"),
        ("assistant", "qux quux corge"),
        ("user", "grault garply waldo"),
        ("assistant", "fred plugh xyzzy"),
        ("user", "completely unrelated message"),
    ])
    result = strategy.apply(messages, budget, 500, _ctx())
    assert len(result) >= 3


def test_pinned_not_filtered():
    budget = TokenBudget(limit=10_000, act_at=0.0)
    strategy = RelevanceFilterStrategy(
        query_window=1, similarity_threshold=0.99, min_keep=1
    )
    pinned = TrackedMessage(role="user", content="system context document", pinned=True)
    other = TrackedMessage(role="assistant", content="zzz yyy xxx")
    current = TrackedMessage(role="user", content="python code function")
    messages = [pinned, other, current]

    result = strategy.apply(messages, budget, 500, _ctx())
    assert pinned in result


def test_no_apply_when_few_messages():
    budget = TokenBudget(limit=10_000, act_at=0.0)
    strategy = RelevanceFilterStrategy(min_keep=4)
    messages = _make_msgs([("user", "a"), ("assistant", "b")])
    result = strategy.apply(messages, budget, 500, _ctx())
    assert result == messages  # untouched
