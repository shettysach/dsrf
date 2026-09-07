"""Visual-only destination settings for the unconstrained motion demo."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PushMotionSettings:
    approach_x: float = 2.0
    goal_x: float = 6.0

    @classmethod
    def from_env(cls) -> "PushMotionSettings":
        defaults = cls()
        return cls(
            approach_x=_float_env("PUSH_MOTION_APPROACH_X", defaults.approach_x),
            goal_x=_float_env("PUSH_MOTION_GOAL_X", defaults.goal_x),
        )

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.approach_x, self.goal_x)):
            raise ValueError("Push-motion waypoints must be finite")
        if not 0.0 < self.approach_x < self.goal_x:
            raise ValueError("Push-motion staging point must lie before its goal")


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)
