from __future__ import annotations

from typing import TYPE_CHECKING

from headroom.core.message import TrackedMessage
from headroom.strategies.base import BaseStrategy, SessionContext

if TYPE_CHECKING:
    from headroom.core.budget import TokenBudget

_SUMMARY_SYSTEM = (
    "You are a conversation summarizer. Produce a concise, factual summary "
    "that preserves key facts, decisions, and important context. Be brief."
)

_SUMMARY_USER_TMPL = (
    "Summarize the following conversation segment in {max_tokens} tokens or fewer:\n\n{messages}"
)


def _format_messages_for_summary(messages: list[TrackedMessage]) -> str:
    parts = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        parts.append(f"{msg.role.upper()}: {content}")
    return "\n\n".join(parts)


class SummarizationStrategy(BaseStrategy):
    """
    Compresses old conversation history into summaries to free up token budget.

    Keeps the most recent `keep_recent` messages verbatim; summarizes everything older.
    """

    priority = 30

    def __init__(
        self,
        keep_recent: int = 6,
        max_summary_tokens: int = 500,
        chunk_size: int = 10,
        summary_model: str | None = None,
    ):
        self.keep_recent = keep_recent
        self.max_summary_tokens = max_summary_tokens
        self.chunk_size = chunk_size
        self.summary_model = summary_model  # if None, uses session model

    def apply(
        self,
        messages: list[TrackedMessage],
        budget: "TokenBudget",
        used_tokens: int,
        ctx: SessionContext,
    ) -> list[TrackedMessage]:
        pinned = [m for m in messages if m.pinned]
        non_pinned = [m for m in messages if not m.pinned]

        if len(non_pinned) <= self.keep_recent:
            return messages  # nothing old enough to summarize

        older = non_pinned[: -self.keep_recent]
        recent = non_pinned[-self.keep_recent :]

        summaries = self._summarize_chunks(older, ctx)
        return pinned + summaries + recent

    def _summarize_chunks(
        self, messages: list[TrackedMessage], ctx: SessionContext
    ) -> list[TrackedMessage]:
        """Split messages into chunks and summarize each one."""
        summaries: list[TrackedMessage] = []
        model = self.summary_model or ctx.model

        for i in range(0, len(messages), self.chunk_size):
            chunk = messages[i : i + self.chunk_size]
            text = _format_messages_for_summary(chunk)
            prompt = _SUMMARY_USER_TMPL.format(
                max_tokens=self.max_summary_tokens, messages=text
            )

            response = ctx.client.messages.create(
                model=model,
                max_tokens=self.max_summary_tokens,
                system=_SUMMARY_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            summary_text = response.content[0].text
            replaced_ids = [m.id for m in chunk]

            summary_msg = TrackedMessage(
                role="user",
                content=f"[SUMMARY of {len(chunk)} messages]: {summary_text}",
                pinned=True,  # summaries are pinned so they won't be re-summarized
                summary_of=replaced_ids,
                token_count=response.usage.output_tokens,
            )
            summaries.append(summary_msg)

        return summaries
