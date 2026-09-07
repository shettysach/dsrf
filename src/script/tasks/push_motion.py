from __future__ import annotations

from dataclasses import dataclass, field

from tasks.push_motion.settings import PushMotionSettings

from shared.messages import AgentCommand, EndEffectorTarget, PushMotionGoal


@dataclass(frozen=True)
class PushMotionScript:
    """Emit one contact-free, feedback-controlled bilateral push motion."""

    prompt: str
    settings: PushMotionSettings = field(default_factory=PushMotionSettings.from_env)

    def next_command(self, observation_id: int) -> AgentCommand | None:
        if observation_id != 0:
            return None
        points = tuple(
            EndEffectorTarget(
                name,
                (
                    self.settings.hand_forward,
                    side * self.settings.hand_half_width,
                    self.settings.hand_height,
                ),
                palm_normal=(1.0, 0.0, 0.0),
            )
            for name, side in (("left_hand", 1), ("right_hand", -1))
        )
        return AgentCommand(
            observation_id=observation_id,
            text=self.prompt,
            motion=self.prompt,
            target_xys=(),
            push_motion_goal=PushMotionGoal(
                approach_xy=(self.settings.approach_x, 0.0),
                target_xy=(self.settings.goal_x, 0.0),
                points=points,
            ),
        )
