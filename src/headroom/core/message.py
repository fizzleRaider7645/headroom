from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

_id_counter = itertools.count(1)


def _next_id() -> int:
    return next(_id_counter)


@dataclass
class TrackedMessage:
    """A conversation message with tracking metadata."""

    role: Literal["user", "assistant"]
    content: str | list[dict]
    id: int = field(default_factory=_next_id)
    token_count: int = 0
    pinned: bool = False
    cache_breakpoint: bool = False
    summary_of: list[int] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def content_hash(self) -> str:
        raw = self.content if isinstance(self.content, str) else str(self.content)
        return hashlib.md5(raw.encode()).hexdigest()

    def to_api_dict(self) -> dict:
        """Return the message in Anthropic SDK format."""
        if isinstance(self.content, str):
            if self.cache_breakpoint:
                return {
                    "role": self.role,
                    "content": [
                        {
                            "type": "text",
                            "text": self.content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            return {"role": self.role, "content": self.content}
        # content is already a list[dict] — may already have cache_control injected
        return {"role": self.role, "content": self.content}

    def to_json(self) -> dict:
        """Serialise to JSON-safe dict for export."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "token_count": self.token_count,
            "pinned": self.pinned,
            "cache_breakpoint": self.cache_breakpoint,
            "summary_of": self.summary_of,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict) -> "TrackedMessage":
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            token_count=data.get("token_count", 0),
            pinned=data.get("pinned", False),
            cache_breakpoint=data.get("cache_breakpoint", False),
            summary_of=data.get("summary_of", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
