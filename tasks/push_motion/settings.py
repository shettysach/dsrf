"""Geometry and pacing for the scripted palm-assisted box push."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from tasks.box_push.settings import BoxPushSettings


@dataclass(frozen=True)
class PushMotionSettings:
    approach_x: float = 2.0
    goal_x: float = 6.0
    navigation_speed: float = 0.4
    push_speed: float = 0.4
    reach_windows: int = 1  # 2
    hand_targets: bool = True
    hand_forward: float = 0.4
    hand_half_width: float = 0.16
    hand_height: float = 0.30
    box_face_gap: float = 0.08
    box_mass: float = 5.0
    vf_enable_distance: float = 0.05
    vf_disable_distance: float = 0.12
    vf_magnitude: float = 15.0
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
            hand_targets=_bool_env("PUSH_MOTION_HANDS", defaults.hand_targets),
            vf_enable_distance=_float_env(
                "PUSH_MOTION_VF_ENABLE_DISTANCE", defaults.vf_enable_distance
            ),
            vf_disable_distance=_float_env(
                "PUSH_MOTION_VF_DISABLE_DISTANCE", defaults.vf_disable_distance
            ),
            vf_magnitude=_float_env("PUSH_MOTION_VF_MAGNITUDE", defaults.vf_magnitude),
        )

    @property
    def box_half_size(self) -> tuple[float, float, float]:
        return BoxPushSettings().half_size

    @property
    def box_start_x(self) -> float:
        return self.approach_x + self.hand_forward + self.box_face_gap + self.box_half_size[0]

    @property
    def box_goal_x(self) -> float:
        return self.goal_x + self.hand_forward + self.box_face_gap + self.box_half_size[0]

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
            self.box_face_gap,
            self.box_mass,
            self.vf_enable_distance,
            self.vf_disable_distance,
            self.vf_magnitude,
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
            or self.box_face_gap < 0.0
            or self.box_mass <= 0.0
            or not 0.0 < self.vf_enable_distance < self.vf_disable_distance
            or self.vf_magnitude <= 0.0
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


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be 'true' or 'false'")
    return value == "true"
