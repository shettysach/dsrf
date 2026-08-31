from __future__ import annotations

from dataclasses import dataclass

from shared.messages import AgentCommand, EndEffectorTarget


@dataclass(frozen=True)
class RightKickScript:
    """Emit one right-leg high-kick command with an optional toe target."""

    prompt: str
    # Robot-local: forward, left, vertical. The target is high and slightly to
    # the right, leaving the supporting left leg unobstructed.
    right_foot_target: tuple[float, float, float] = (0.65, -0.15, 0.65)

    def next_command(self, observation_id: int) -> AgentCommand | None:
        if observation_id != 0:
            return None
        return AgentCommand(
            observation_id=observation_id,
            text=self.prompt,
            motion=self.prompt,
            target_xys=(),
            end_effectors=(
                EndEffectorTarget("right_foot", self.right_foot_target),
            ),
        )
