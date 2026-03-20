from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from headroom.core.budget import TokenBudget


@pytest.fixture
def mock_client():
    """A mock anthropic.Anthropic client for testing without real API calls."""
    client = MagicMock()

    # Default count_tokens response
    count_response = MagicMock()
    count_response.input_tokens = 100
    client.messages.count_tokens.return_value = count_response

    # Default messages.create response
    content_block = MagicMock()
    content_block.text = "Hello! I'm the assistant."
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 20
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    msg_response = MagicMock()
    msg_response.content = [content_block]
    msg_response.usage = usage
    client.messages.create.return_value = msg_response

    # Default messages.stream response (context manager)
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    stream_ctx.text_stream = ["Hello", "! I'm", " the assistant."]
    stream_final = MagicMock()
    stream_final.content = [content_block]
    stream_final.usage = usage
    stream_ctx.get_final_message = MagicMock(return_value=stream_final)
    client.messages.stream.return_value = stream_ctx

    return client


@pytest.fixture
def small_budget():
    return TokenBudget(limit=1_000, warn_at=0.5, act_at=0.7, reserve=50)


@pytest.fixture
def normal_budget():
    return TokenBudget(limit=200_000)
