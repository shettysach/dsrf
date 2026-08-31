from __future__ import annotations

from dataclasses import dataclass

from shared.messages import AgentCommand


@dataclass(frozen=True)
class PromptScript:
    """Emit one unconstrained ARDY motion prompt."""

    prompt: str

    def next_command(self, observation_id: int) -> AgentCommand | None:
        if observation_id != 0:
            return None
        return AgentCommand(
            observation_id=observation_id,
            text=self.prompt,
            motion=self.prompt,
            target_xys=(),
        )
