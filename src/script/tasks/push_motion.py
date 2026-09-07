from __future__ import annotations

from dataclasses import dataclass, field

from tasks.push_motion.settings import PushMotionSettings

from shared.messages import AgentCommand, RootPathGoal


@dataclass(frozen=True)
class PushMotionScript:
    """Emit a plain text motion with only 2D root waypoints."""

    prompt: str
    settings: PushMotionSettings = field(default_factory=PushMotionSettings.from_env)

    def next_command(self, observation_id: int) -> AgentCommand | None:
        if observation_id != 0:
            return None
        return AgentCommand(
            observation_id=observation_id,
            text=self.prompt,
            motion=self.prompt,
            target_xys=(),
            root_path_goal=RootPathGoal(
                approach_xy=(self.settings.approach_x, 0.0),
                target_xy=(self.settings.goal_x, 0.0),
            ),
        )
