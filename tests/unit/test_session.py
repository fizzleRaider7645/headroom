from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from headroom.core.budget import TokenBudget
from headroom.core.session import Session


@pytest.fixture
def session(mock_client):
    with patch("headroom.core.session.anthropic.Anthropic", return_value=mock_client):
        s = Session(model="claude-opus-4-6", budget=TokenBudget(limit=10_000))
        # Patch the internal client
        s._client = mock_client
        s._counter._client = mock_client
        return s


@pytest.mark.asyncio
async def test_send_appends_messages(session):
    assert len(session.history) == 0
    await session.send("Hello")
    assert len(session.history) == 2  # user + assistant
    assert session.history[0].role == "user"
    assert session.history[1].role == "assistant"


@pytest.mark.asyncio
async def test_send_increments_turns(session):
    await session.send("Hello")
    await session.send("World")
    assert session.token_usage.turns == 2


@pytest.mark.asyncio
async def test_pin_prevents_trim(session, mock_client):
    # Set budget so overflow triggers
    session.budget = TokenBudget(limit=100, warn_at=0.1, act_at=0.2, reserve=0)
    mock_client.messages.count_tokens.return_value.input_tokens = 99

    await session.send("First message")
    msg_id = session.history[0].id
    session.pin(msg_id)

    # Even if BudgetGuard runs, pinned message survives
    from headroom.strategies.budget import BudgetGuardStrategy
    strategy = BudgetGuardStrategy()
    msgs = session._messages
    result = strategy.apply(msgs, session.budget, 99, MagicMock())
    pinned = [m for m in result if m.pinned]
    assert any(m.id == msg_id for m in pinned)


def test_add_context(session):
    msg = session.add_context("Important document content", pinned=True)
    assert msg.pinned is True
    assert msg in session.history


@pytest.mark.asyncio
async def test_export_and_load(session, tmp_path):
    await session.send("Hello!")
    export_path = tmp_path / "session.json"
    session.export(export_path)

    assert export_path.exists()
    data = json.loads(export_path.read_text())
    assert data["model"] == "claude-opus-4-6"
    assert len(data["messages"]) == 2

    with patch("headroom.core.session.anthropic.Anthropic", return_value=session._client):
        restored = Session.load(export_path)
    assert len(restored.history) == 2
    assert restored.model == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_disabled_strategy_is_skipped(session, mock_client):
    """A strategy with enabled=False must not run, even when budget conditions are met."""
    session.budget = TokenBudget(limit=100, warn_at=0.1, act_at=0.2, reserve=0)
    mock_client.messages.count_tokens.return_value.input_tokens = 99

    # Seed two messages so BudgetGuard has something to drop
    await session.send("First")
    await session.send("Second")
    assert len(session.history) == 4  # 2 user + 2 assistant

    # Disable BudgetGuardStrategy
    for s in session._strategies:
        if s.name == "BudgetGuardStrategy":
            s.enabled = False

    # Force overflow conditions on next send
    mock_client.messages.count_tokens.return_value.input_tokens = 101
    before_count = len(session.history)

    await session.send("Third")

    # BudgetGuard was disabled — it must not have dropped any messages
    # (history grows by 2: new user + assistant)
    assert len(session.history) == before_count + 2


@pytest.mark.asyncio
async def test_clear_resets_messages_and_stats(session):
    """Session.clear() must wipe history, counter cache, and all stats."""
    await session.send("Hello")
    await session.send("World")
    assert len(session.history) == 4
    assert session.token_usage.turns == 2

    session.clear()

    assert len(session.history) == 0
    assert session.token_usage.turns == 0
    assert session.token_usage.used == 0
    assert session.token_usage.cache_hits == 0
    assert len(session._counter._cache) == 0


@pytest.mark.asyncio
async def test_clear_allows_fresh_send(session):
    """After clear(), sending a message works normally."""
    await session.send("Before clear")
    session.clear()
    await session.send("After clear")
    assert len(session.history) == 2  # only the post-clear exchange
    assert session.history[0].content == "After clear"


@pytest.mark.asyncio
async def test_warning_callback_fires(session, mock_client):
    session.budget = TokenBudget(limit=100, warn_at=0.5, act_at=0.9, reserve=0)
    mock_client.messages.count_tokens.return_value.input_tokens = 60

    events = []
    session.on_warning = events.append

    await session.send("test")
    assert len(events) == 1
    assert events[0].status in ("warn", "act", "overflow")
