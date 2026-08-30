from __future__ import annotations

from typing import Protocol

from shared.messages import AgentCommand


class TaskScript(Protocol):
    """Emit the command for an observation, or stop issuing commands."""

    def next_command(self, observation_id: int) -> AgentCommand | None: ...
