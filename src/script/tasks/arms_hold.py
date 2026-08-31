from __future__ import annotations

from dataclasses import dataclass

from shared.messages import AgentCommand, EndEffectorTarget


@dataclass(frozen=True)
class ArmsHoldScript:
    """Emit one static, shoulder-height bilateral arm-hold command."""

    prompt: str
    # Robot-local: forward, left, vertical. With the G1 standing root, these
    # place both hands near the natural shoulder-height arms-forward pose.
    hand_targets: tuple[tuple[float, float, float], ...] = (
        (0.58, 0.16, 0.40),
        (0.58, -0.16, 0.40),
    )

    def next_command(self, observation_id: int) -> AgentCommand | None:
        if observation_id != 0:
            return None
        return AgentCommand(
            observation_id=observation_id,
            text=self.prompt,
            motion=self.prompt,
            target_xys=(),
            end_effectors=(
                EndEffectorTarget("left_hand", self.hand_targets[0]),
                EndEffectorTarget("right_hand", self.hand_targets[1]),
            ),
        )
