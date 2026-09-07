"""Visual-only destination settings for the unconstrained motion demo."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PushMotionSettings:
    goal_x: float = 6.0

    @classmethod
    def from_env(cls) -> "PushMotionSettings":
        defaults = cls()
        return cls(goal_x=_float_env("PUSH_MOTION_GOAL_X", defaults.goal_x))

    def __post_init__(self) -> None:
        if not math.isfinite(self.goal_x):
            raise ValueError("Push-motion goal must be finite")


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)
