from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from headroom.core.budget import TokenBudget
from headroom.core.session import Session
from headroom.dashboard.app import create_app
from headroom.dashboard.state import get_state


@pytest.fixture(autouse=True)
def setup_state(mock_client):
    state = get_state()
    with patch("headroom.core.session.anthropic.Anthropic", return_value=mock_client):
        session = Session(model="claude-opus-4-6", budget=TokenBudget(limit=10_000))
        session._client = mock_client
        session._counter._client = mock_client
    state.session = session
    state.session_name = "test-session"
    yield
    state.session = None
    state.event_log.clear()


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_get_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "headroom" in response.text.lower()


def test_api_status_returns_usage(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "headroom" in data
    assert "used" in data
    assert "limit" in data
    assert data["model"] == "claude-opus-4-6"


def test_api_messages_empty_initially(client):
    response = client.get("/api/messages")
    assert response.status_code == 200
    assert response.json() == []


def test_budget_bar_partial(client):
    response = client.get("/partials/budget_bar")
    assert response.status_code == 200
    assert "headroom" in response.text.lower() or "tokens" in response.text.lower()


def test_strategy_toggle(client):
    response = client.post("/api/strategy/SummarizationStrategy/toggle")
    assert response.status_code == 200
    data = response.json()
    assert "enabled" in data


def test_strategy_toggle_actually_disables_strategy(client):
    """Toggling a strategy must set strategy.enabled=False on the live instance."""
    state = get_state()

    # Confirm enabled by default
    strat = next(
        s for s in state.session._strategies if s.name == "SummarizationStrategy"
    )
    assert strat.enabled is True

    # Toggle off
    client.post("/api/strategy/SummarizationStrategy/toggle")
    assert strat.enabled is False

    # Toggle back on
    client.post("/api/strategy/SummarizationStrategy/toggle")
    assert strat.enabled is True


def test_strategy_toggle_htmx_partial(client):
    """HTMX toggle endpoint returns HTML card with correct enabled state."""
    response = client.post("/partials/strategy/SummarizationStrategy/toggle")
    assert response.status_code == 200
    # After one toggle it should be disabled — card should reflect that
    assert "OFF" in response.text or "disabled" in response.text.lower() or "false" in response.text.lower()


def test_get_api_strategies(client):
    """GET /api/strategies returns list of strategy dicts."""
    response = client.get("/api/strategies")
    assert response.status_code == 200
    strategies = response.json()
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    names = {s["name"] for s in strategies}
    assert "BudgetGuardStrategy" in names
    assert "CacheInjectionStrategy" in names
    for s in strategies:
        assert "name" in s
        assert "priority" in s
        assert "enabled" in s
        assert "params" in s


def test_clear_chat_wipes_session(client):
    """POST /partials/clear must clear session history and return empty HTML."""
    state = get_state()
    # Seed a message directly so we don't need a real API call
    from headroom.core.message import TrackedMessage
    state.session._messages.append(TrackedMessage(role="user", content="seed"))
    assert len(state.session.history) == 1

    response = client.post("/partials/clear")

    assert response.status_code == 200
    assert response.text == ""           # empty HTML → message list goes blank
    assert len(state.session.history) == 0
    assert state.session.token_usage.turns == 0


def test_send_partial_returns_only_assistant_message(client):
    response = client.post("/partials/send", data={"message": "duplicate me"})

    assert response.status_code == 200
    assert "message-assistant" in response.text
    assert "message-user" not in response.text
    assert "duplicate me" not in response.text
