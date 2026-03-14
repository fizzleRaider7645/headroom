from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from headroom.core.session import Session


@dataclass
class DashboardState:
    session: "Session | None" = None
    session_name: str = "untitled"
    event_log: list[dict] = field(default_factory=list)

    # Per-strategy enabled flags (name -> bool)
    strategy_enabled: dict[str, bool] = field(default_factory=dict)

    def log_event(self, event_type: str, **kwargs) -> None:
        from datetime import datetime, timezone
        self.event_log.append(
            {"type": event_type, "ts": datetime.now(timezone.utc).isoformat(), **kwargs}
        )


_state = DashboardState()


def get_state() -> DashboardState:
    return _state
