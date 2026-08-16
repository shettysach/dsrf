"""Human-readable, opt-in VLM diagnostics for the direct agent client."""

from __future__ import annotations

import json
from typing import Any


class AgentDebug:
    """Log reasoning and commands without exposing observation image bytes."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def response(
        self,
        node: Any,
        *,
        observation_id: int,
        reasoning: str | None,
        command: str,
    ) -> None:
        if not self.enabled:
            return
        node.log(
            "warn",
            f"[OBS {observation_id}] VLM reasoning:\n{reasoning or '(none returned)'}\n"
            f"[OBS {observation_id}] VLM command:\n{_pretty_command(command)}",
            target="dsrf.agent.debug",
            fields={
                "event": "agent_debug_response",
                "observation_id": str(observation_id),
            },
        )


def _pretty_command(command: str) -> str:
    try:
        return json.dumps(json.loads(command), indent=2, sort_keys=True)
    except json.JSONDecodeError:
        return command
