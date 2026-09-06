from __future__ import annotations

from dataclasses import dataclass, field

from tasks.box_push.settings import BoxPushSettings

from shared.messages import AgentCommand, ContactGoal, EndEffectorTarget


@dataclass(frozen=True)
class PushScript:
    """One logical request; the sim executes approach, contact, and push."""

    prompt: str
    settings: BoxPushSettings = field(default_factory=BoxPushSettings.from_env)

    def next_command(self, observation_id: int) -> AgentCommand | None:
        if observation_id != 0:
            return None
        depth, _, height = self.settings.half_size
        return AgentCommand(
            observation_id=observation_id,
            text=self.prompt,
            motion=self.prompt,
            target_xys=(),
            contact_goal=ContactGoal(
                body="box",
                target_xy=(self.settings.goal_x, 0.0),
                points=tuple(
                    EndEffectorTarget(name, (-depth, side * 0.18, 0.60 - height))
                    for name, side in (("left_hand", 1), ("right_hand", -1))
                ),
                goal_half_size=self.settings.goal_half_size,
            ),
        )
