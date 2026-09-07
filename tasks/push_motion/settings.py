"""Defaults for the contact-free scripted push motion."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PushMotionSettings:
    """Pacing and bilateral palm pose for a virtual straight push."""

    approach_x: float = 2.0
    goal_x: float = 6.0
    hand_forward: float = 0.58
    hand_half_width: float = 0.16
    hand_height: float = 0.40
    navigation_speed: float = 0.4
    push_speed: float = 0.15
    reach_windows: int = 1
    timeout: float = 90.0
    stall_seconds: float = 4.0
    progress_distance: float = 0.02
    approach_prompt: str = "walk forward"
    reach_prompt: str = (
        "stand and extend both arms straight forward, palms facing forward"
    )
    push_prompt: str = (
        "walk forward with both arms straight forward, palms facing forward"
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
            self.hand_forward,
            self.hand_half_width,
            self.hand_height,
            self.navigation_speed,
            self.push_speed,
            self.timeout,
            self.stall_seconds,
            self.progress_distance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Push-motion settings must be finite")
        if not 0.0 < self.approach_x < self.goal_x:
            raise ValueError("Approach point must lie between the start and goal")
        if not all(
            value > 0.0
            for value in (
                self.hand_forward,
                self.hand_half_width,
                self.navigation_speed,
                self.push_speed,
                self.timeout,
                self.stall_seconds,
                self.progress_distance,
            )
        ):
            raise ValueError("Push-motion dimensions and pacing must be positive")
        if self.reach_windows < 1:
            raise ValueError("Push-motion reach windows must be positive")
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
