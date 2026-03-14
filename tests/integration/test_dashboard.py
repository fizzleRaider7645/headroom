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
