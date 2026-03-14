from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anthropic

from headroom.core.budget import TokenBudget, TokenUsage
from headroom.core.message import TrackedMessage
from headroom.counting.counter import TokenCounter
from headroom.strategies.base import BaseStrategy, SessionContext
from headroom.strategies import default_strategies


@dataclass
class BudgetEvent:
    status: str
    used: int
    limit: int
    headroom: int


@dataclass
class TrimEvent:
    strategy: str
    dropped: int
    before: int
    after: int


class Session:
    """
    Primary interface for managing a Claude conversation with automatic
    context optimization.

    Usage::

        session = Session(model="claude-opus-4-6")
        response = await session.send("Hello!")
        print(session.token_usage.headroom)
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        api_key: str | None = None,
        system: str | None = None,
        budget: TokenBudget | None = None,
        strategies: list[BaseStrategy] | None = None,
        on_warning: Callable[[BudgetEvent], None] | None = None,
        on_trim: Callable[[TrimEvent], None] | None = None,
        auto_cache: bool = True,
        max_tokens: int = 1_024,
    ):
        self.model = model
        self.system = system
        self.budget = budget or TokenBudget.for_model(model)
        self.max_tokens = max_tokens
        self.on_warning = on_warning
        self.on_trim = on_trim

        self._client = anthropic.Anthropic(api_key=api_key)
        self._counter = TokenCounter(self._client, model)
        self._strategies: list[BaseStrategy] = (
            strategies if strategies is not None else default_strategies()
        )
        self._messages: list[TrackedMessage] = []
        self._last_sent_ids: frozenset[int] = frozenset()

        # Usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        self._cache_hits = 0
        self._turns = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self,
        user_message: str | list[dict],
        *,
        extra_params: dict | None = None,
    ) -> anthropic.types.Message:
        """Send a user message and return the assistant response."""
        user_msg = TrackedMessage(role="user", content=user_message)
        self._messages.append(user_msg)

        try:
            return await self._send_inner(user_msg, extra_params=extra_params)
        except Exception:
            # Roll back the user message so history stays clean on failure
            if user_msg in self._messages:
                self._messages.remove(user_msg)
            raise

    async def _send_inner(
        self,
        user_msg: "TrackedMessage",
        *,
        extra_params: dict | None = None,
    ) -> anthropic.types.Message:
        # 1. Exact token count
        used = self._counter.count_exact(self._messages, system=self.system)
        self._update_message_counts()

        # 2. Check budget and fire warning
        status = self.budget.status(used)
        if status in ("warn", "act", "overflow") and self.on_warning:
            self.on_warning(
                BudgetEvent(
                    status=status,
                    used=used,
                    limit=self.budget.limit,
                    headroom=self.budget.headroom(used),
                )
            )

        # 3. Run strategy pipeline
        optimized = await self._apply_strategies(self._messages, used)

        # 4. Build API call arguments
        api_messages = [m.to_api_dict() for m in optimized]
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }
        if self.system:
            kwargs["system"] = self.system
        if extra_params:
            kwargs.update(extra_params)

        # 5. Call the API
        response = self._client.messages.create(**kwargs)

        # 6. Track usage
        usage = response.usage
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self._cache_read_tokens += cache_read
        self._cache_write_tokens += cache_write
        if cache_read > 0:
            self._cache_hits += 1
        self._turns += 1

        # 7. Append assistant response to history
        assistant_text = response.content[0].text if response.content else ""
        assistant_msg = TrackedMessage(
            role="assistant",
            content=assistant_text,
            token_count=usage.output_tokens,
        )
        self._messages.append(assistant_msg)
        self._last_sent_ids = frozenset(m.id for m in optimized)

        return response

    def send_sync(
        self,
        user_message: str | list[dict],
        **kwargs,
    ) -> anthropic.types.Message:
        """Synchronous wrapper around send()."""
        return asyncio.run(self.send(user_message, **kwargs))

    def pin(self, message_id: int) -> None:
        """Mark a message as pinned (immune to trimming)."""
        for msg in self._messages:
            if msg.id == message_id:
                msg.pinned = True
                return
        raise ValueError(f"No message with id={message_id}")

    def add_context(self, text: str, *, pinned: bool = True) -> TrackedMessage:
        """Inject a static context chunk (e.g., a document) as a user message."""
        msg = TrackedMessage(role="user", content=text, pinned=pinned)
        self._messages.append(msg)
        return msg

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def token_usage(self) -> TokenUsage:
        used = self._counter.count_sum(self._messages)
        return TokenUsage(
            used=used,
            limit=self.budget.limit,
            reserve=self.budget.reserve,
            cache_read_tokens=self._cache_read_tokens,
            cache_write_tokens=self._cache_write_tokens,
            cache_hits=self._cache_hits,
            turns=self._turns,
        )

    @property
    def history(self) -> list[TrackedMessage]:
        return list(self._messages)

    def messages_for_api(self) -> list[dict]:
        return [m.to_api_dict() for m in self._messages]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export(self, path: str | Path) -> None:
        """Export session to JSON."""
        data = {
            "model": self.model,
            "system": self.system,
            "budget": {
                "limit": self.budget.limit,
                "warn_at": self.budget.warn_at,
                "act_at": self.budget.act_at,
                "reserve": self.budget.reserve,
            },
            "messages": [m.to_json() for m in self._messages],
            "stats": {
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "cache_read_tokens": self._cache_read_tokens,
                "cache_write_tokens": self._cache_write_tokens,
                "cache_hits": self._cache_hits,
                "turns": self._turns,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "Session":
        """Restore a session from an exported JSON file."""
        data = json.loads(Path(path).read_text())
        budget_data = data.get("budget", {})
        budget = TokenBudget(**budget_data) if budget_data else None
        session = cls(
            model=data["model"],
            system=data.get("system"),
            budget=budget,
            **kwargs,
        )
        session._messages = [TrackedMessage.from_json(m) for m in data["messages"]]
        stats = data.get("stats", {})
        session._total_input_tokens = stats.get("total_input_tokens", 0)
        session._total_output_tokens = stats.get("total_output_tokens", 0)
        session._cache_read_tokens = stats.get("cache_read_tokens", 0)
        session._cache_write_tokens = stats.get("cache_write_tokens", 0)
        session._cache_hits = stats.get("cache_hits", 0)
        session._turns = stats.get("turns", 0)
        return session

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _apply_strategies(
        self, messages: list[TrackedMessage], used_tokens: int
    ) -> list[TrackedMessage]:
        ctx = SessionContext(
            client=self._client, model=self.model, system=self.system
        )
        result = list(messages)

        for strategy in sorted(self._strategies, key=lambda s: s.priority):
            if strategy.should_apply(self.budget, used_tokens):
                before = len(result)
                # Summarizer makes API calls; run in thread pool to stay async-safe
                if asyncio.iscoroutinefunction(strategy.apply):
                    result = await strategy.apply(result, self.budget, used_tokens, ctx)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, strategy.apply, result, self.budget, used_tokens, ctx
                    )
                after = len(result)
                if self.on_trim and before != after:
                    self.on_trim(
                        TrimEvent(
                            strategy=strategy.name,
                            dropped=before - after,
                            before=before,
                            after=after,
                        )
                    )
                # Recalculate after each strategy
                used_tokens = self._counter.count_sum(result)

        return result

    def _update_message_counts(self) -> None:
        """Update token_count on each message using cached estimates."""
        for msg in self._messages:
            if msg.token_count == 0:
                msg.token_count = self._counter._estimate_one(msg)
