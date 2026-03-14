from unittest.mock import MagicMock

from headroom.core.budget import TokenBudget
from headroom.core.message import TrackedMessage
from headroom.strategies.base import SessionContext
from headroom.strategies.summarizer import SummarizationStrategy


def _mock_ctx(summary_text: str = "Summary of conversation.") -> SessionContext:
    client = MagicMock()
    content = MagicMock()
    content.text = summary_text
    usage = MagicMock()
    usage.output_tokens = 10
    resp = MagicMock()
    resp.content = [content]
    resp.usage = usage
    client.messages.create.return_value = resp
    return SessionContext(client=client, model="claude-opus-4-6")


def _make_msgs(n: int) -> list[TrackedMessage]:
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(TrackedMessage(role=role, content=f"Message {i}"))
    return msgs


def test_summarizes_old_messages():
    budget = TokenBudget(limit=10_000, act_at=0.0)
    strategy = SummarizationStrategy(keep_recent=2, chunk_size=10)
    ctx = _mock_ctx()

    messages = _make_msgs(8)
    result = strategy.apply(messages, budget, 500, ctx)

    # Should have: 1 summary + 2 recent = 3
    assert len(result) == 3
    # The summary message should reference the replaced IDs
    summary = result[0]
    assert summary.pinned is True
    assert len(summary.summary_of) == 6  # 8 - 2 recent


def test_pinned_not_summarized():
    budget = TokenBudget(limit=10_000, act_at=0.0)
    strategy = SummarizationStrategy(keep_recent=2)
    ctx = _mock_ctx()

    pinned = TrackedMessage(role="user", content="Pinned context", pinned=True)
    others = _make_msgs(6)
    messages = [pinned] + others

    result = strategy.apply(messages, budget, 500, ctx)
    # Pinned message must survive
    assert pinned in result


def test_too_few_messages_returns_unchanged():
    budget = TokenBudget(limit=10_000, act_at=0.0)
    strategy = SummarizationStrategy(keep_recent=6)
    ctx = _mock_ctx()

    messages = _make_msgs(4)  # less than keep_recent non-pinned
    result = strategy.apply(messages, budget, 500, ctx)
    assert result == messages  # unchanged


def test_summary_marked_pinned():
    budget = TokenBudget(limit=10_000, act_at=0.0)
    strategy = SummarizationStrategy(keep_recent=1, chunk_size=5)
    ctx = _mock_ctx("A brief summary.")

    messages = _make_msgs(6)
    result = strategy.apply(messages, budget, 500, ctx)

    for msg in result:
        if msg.summary_of:
            assert msg.pinned is True
