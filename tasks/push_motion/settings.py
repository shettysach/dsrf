"""Visual-only destination settings for the unconstrained motion demo."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PushMotionSettings:
    approach_x: float = 2.0
    goal_x: float = 6.0
    navigation_speed: float = 0.4
    push_speed: float = 0.4
    reach_windows: int = 1
    hand_forward: float = 0.58
    hand_half_width: float = 0.16
    hand_height: float = 0.20
    timeout: float = 90.0
    approach_prompt: str = "Walking forward"
    reach_prompt: str = (
        "A person reaches out forwards, fully extending arms to push a box"
    )
    push_prompt: str = (
        "A person walks forwards with both arms fully extended to push a box"
    )

    @classmethod
    def from_env(cls) -> "PushMotionSettings":
        defaults = cls()
        return cls(
            approach_x=_float_env("PUSH_MOTION_APPROACH_X", defaults.approach_x),
            goal_x=_float_env("PUSH_MOTION_GOAL_X", defaults.goal_x),
            navigation_speed=_float_env(
                "PUSH_MOTION_NAVIGATION_SPEED", defaults.navigation_speed
            ),
            push_speed=_float_env("PUSH_MOTION_SPEED", defaults.push_speed),
            reach_windows=_int_env("PUSH_MOTION_REACH_WINDOWS", defaults.reach_windows),
        )

    def __post_init__(self) -> None:
        values = (
            self.approach_x,
            self.goal_x,
            self.navigation_speed,
            self.push_speed,
            self.timeout,
            self.hand_forward,
            self.hand_half_width,
            self.hand_height,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Push-motion waypoints must be finite")
        if not 0.0 < self.approach_x < self.goal_x:
            raise ValueError("Push-motion staging point must lie before its goal")
        if (
            not all(
                value > 0.0
                for value in (
                    self.navigation_speed,
                    self.push_speed,
                    self.timeout,
                    self.hand_forward,
                    self.hand_half_width,
                )
            )
            or self.reach_windows < 1
        ):
            raise ValueError("Push-motion pacing must be positive")
        if not all(
            prompt.strip()
            for prompt in (self.approach_prompt, self.reach_prompt, self.push_prompt)
        ):
            raise ValueError("Push-motion prompts must be non-empty")


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)
