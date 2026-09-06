"""Lightweight scene settings shared by the script and simulator."""

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BoxPushSettings:
    box_x: float = 3.0
    goal_x: float = 6.0
    half_size: tuple[float, float, float] = (0.4, 0.4, 0.4)
    goal_half_size: float = 0.65

    @classmethod
    def from_env(cls) -> "BoxPushSettings":
        return cls(
            box_x=float(os.environ.get("BOX_PUSH_START_X", "3.0")),
            goal_x=float(os.environ.get("BOX_PUSH_GOAL_X", "6.0")),
        )

    def __post_init__(self) -> None:
        if not math.isfinite(self.box_x) or not math.isfinite(self.goal_x):
            raise ValueError("Box and goal positions must be finite")
        if self.box_x <= 1.0 or self.goal_x <= self.box_x:
            raise ValueError(
                "Box must start ahead of the robot, with the goal beyond it"
            )
