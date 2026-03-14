from __future__ import annotations

import anthropic

from headroom.core.message import TrackedMessage


class TokenCounter:
    """Counts tokens using the Anthropic SDK with a local cache."""

    def __init__(self, client: anthropic.Anthropic, model: str):
        self._client = client
        self._model = model
        # (id, content_hash) -> token_count
        self._cache: dict[tuple[int, str], int] = {}

    def count_exact(
        self,
        messages: list[TrackedMessage],
        system: str | None = None,
    ) -> int:
        """
        Exact count via client.messages.count_tokens().
        Called once per send() call. Updates per-message cache.
        """
        api_messages = [m.to_api_dict() for m in messages]
        kwargs: dict = {"model": self._model, "messages": api_messages}
        if system:
            kwargs["system"] = system

        response = self._client.messages.count_tokens(**kwargs)
        total: int = response.input_tokens

        # Update individual message cache with estimated split
        # (exact individual counts aren't available from the API, so we estimate)
        for msg in messages:
            key = (msg.id, msg.content_hash())
            if key not in self._cache:
                self._cache[key] = self._estimate_one(msg)

        return total

    def count_sum(self, messages: list[TrackedMessage]) -> int:
        """
        Sum of cached individual estimates. Fast, no API call.
        Used for live UI updates and post-strategy recalculation.
        """
        return sum(self._get_or_estimate(m) for m in messages)

    def _get_or_estimate(self, msg: TrackedMessage) -> int:
        key = (msg.id, msg.content_hash())
        if key not in self._cache:
            self._cache[key] = self._estimate_one(msg)
        return self._cache[key]

    def _estimate_one(self, msg: TrackedMessage) -> int:
        """Fast heuristic: ~4 chars per token."""
        if isinstance(msg.content, str):
            return max(1, len(msg.content) // 4)
        # list[dict] content
        total = 0
        for block in msg.content:
            text = block.get("text", "") or str(block)
            total += max(1, len(text) // 4)
        return total

    def update_from_api(self, msg: TrackedMessage, exact_count: int) -> None:
        """Store an exact token count for a single message (e.g. from usage response)."""
        key = (msg.id, msg.content_hash())
        self._cache[key] = exact_count
        msg.token_count = exact_count

    def estimate_text(self, text: str) -> int:
        """Estimate tokens for arbitrary text without a TrackedMessage."""
        return max(1, len(text) // 4)
