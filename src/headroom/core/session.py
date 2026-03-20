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
from headroom.strategies import default_strategies
from headroom.strategies.base import BaseStrategy, SessionContext
from headroom.strategies.cache import CacheInjectionStrategy


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
        if not auto_cache:
            for s in self._strategies:
                if isinstance(s, CacheInjectionStrategy):
                    s.enabled = False
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

    async def stream(
        self,
        user_message: str | list[dict],
        *,
        extra_params: dict | None = None,
    ):
        """Stream the assistant response as an async generator of text chunks.

        Usage::

            async for chunk in session.stream("Hello!"):
                print(chunk, end="", flush=True)
        """
        user_msg = TrackedMessage(role="user", content=user_message)
        self._messages.append(user_msg)  # noqa: must happen before try so rollback works
        try:
            async for chunk in self._stream_inner(user_msg, extra_params=extra_params):
                yield chunk
        except Exception:
            if user_msg in self._messages:
                self._messages.remove(user_msg)
            raise

    async def _stream_inner(
        self,
        user_msg: "TrackedMessage",
        *,
        extra_params: dict | None = None,
    ):
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

        # 5. Stream via thread pool (sync Anthropic client)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        final_holder: list = []
        error_holder: list = []

        def _run_stream() -> None:
            try:
                with self._client.messages.stream(**kwargs) as s:
                    for text in s.text_stream:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
                    final_holder.append(s.get_final_message())
            except Exception as exc:
                error_holder.append(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _run_stream)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

        if error_holder:
            raise error_holder[0]

        # 6. Track usage
        final = final_holder[0]
        usage = final.usage
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
        assistant_text = final.content[0].text if final.content else ""
        assistant_msg = TrackedMessage(
            role="assistant",
            content=assistant_text,
            token_count=usage.output_tokens,
        )
        self._messages.append(assistant_msg)
        self._last_sent_ids = frozenset(m.id for m in optimized)

    def stream_sync(
        self,
        user_message: str | list[dict],
        *,
        extra_params: dict | None = None,
    ):
        """Synchronous generator that streams the assistant response chunk by chunk.

        Usage::

            for chunk in session.stream_sync("Hello!"):
                print(chunk, end="", flush=True)
        """
        import queue as _stdlib_queue
        import threading

        q: _stdlib_queue.Queue = _stdlib_queue.Queue()
        exc_holder: list = []

        async def _run() -> None:
            try:
                async for chunk in self.stream(user_message, extra_params=extra_params):
                    q.put(chunk)
            except Exception as exc:
                exc_holder.append(exc)
            finally:
                q.put(None)

        t = threading.Thread(target=lambda: asyncio.run(_run()), daemon=True)
        t.start()

        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield chunk

        t.join()
        if exc_holder:
            raise exc_holder[0]

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

    def clear(self) -> None:
        """Clear conversation history, preserving session-level stats.

        Wipes all messages and the token-count cache so the context window is
        empty and headroom returns to its maximum.  Cumulative counters
        (turns, cache hits, total tokens) are intentionally kept so the
        dashboard stats reflect the full session, not just the current chat.

        The session configuration (model, system, budget, strategies) and the
        API client are unchanged.
        """
        self._messages.clear()
        self._counter.clear()
        self._last_sent_ids = frozenset()

    def reset_stats(self) -> None:
        """Reset all cumulative session statistics (turns, token totals, etc).

        Call this when you want a completely fresh start, including zeroing
        out the stats shown in the dashboard budget bar.
        """
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        self._cache_hits = 0
        self._turns = 0

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
        ctx = SessionContext(client=self._client, model=self.model, system=self.system)
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
