"""Single source of defaults and optional environment overrides for box push."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BoxPushSettings:
    """Geometry, ARDY constraints, pacing, and assistance for scripted push."""

    # Scene geometry.
    box_x: float = 3.0
    goal_x: float = 6.0
    half_size: tuple[float, float, float] = (0.5, 0.5, 0.5)
    box_mass: float = 0.5
    box_friction: tuple[float, float, float] = (0.2, 0.01, 0.001)
    box_slide_damping: float = 0.8
    goal_half_size: float = 0.65

    # Box-relative palm targets. ``hand_target_world_z`` is intentionally
    # world-space so it is easy to tune against the visible box height.
    hand_half_width: float = 0.30
    hand_target_world_z: float = 0.75
    # Semantic robot-local direction (forward, left, up) for both palms.
    palm_normal: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # Per-phase ARDY priors and temporal pacing.
    approach_prompt: str = "walk forward"
    contact_prompt: str = "stand and extend both arms forward, placing both palms against the box"
    push_prompt: str = "walk forward with both hands held forward"
    settle_prompt: str = "stand"
    navigation_speed: float = 0.4
    push_speed: float = 0.15
    standoff: float = 0.50
    contact_windows: int = 2
    contact_dwell: float = 0.2
    contact_loss: float = 0.1
    reacquisitions: int = 2
    timeout: float = 90.0
    stall_seconds: float = 4.0
    progress_distance: float = 0.02

    # Optional contact assistance.
    weld_enabled: bool = False
    weld_solref: tuple[float, float] = (0.1, 1.0)
    virtual_force_magnitude: float = 15.0
    virtual_force_max: float = 30.0

    @classmethod
    def from_env(cls) -> "BoxPushSettings":
        defaults = cls()
        return cls(
            box_x=_float_env("BOX_PUSH_START_X", defaults.box_x),
            goal_x=_float_env("BOX_PUSH_GOAL_X", defaults.goal_x),
            navigation_speed=_float_env(
                "PUSH_NAVIGATION_SPEED", defaults.navigation_speed
            ),
            push_speed=_float_env("PUSH_SPEED", defaults.push_speed),
            standoff=_float_env("PUSH_STANDOFF", defaults.standoff),
            contact_windows=_int_env("PUSH_CONTACT_WINDOWS", defaults.contact_windows),
            weld_enabled=_bool_env("PUSH_WELD", defaults.weld_enabled),
        )

    @property
    def box_top_z(self) -> float:
        """World height of the top face when the box rests on the floor."""
        return 2.0 * self.half_size[2]

    @property
    def hand_target_local_z(self) -> float:
        """Target height relative to the box center, as ARDY expects."""
        return self.hand_target_world_z - self.half_size[2]

    def prompt_for_phase(self, phase: str) -> str:
        return {
            "approach": self.approach_prompt,
            "contact": self.contact_prompt,
            "push": self.push_prompt,
            "settle": self.settle_prompt,
        }[phase]

    def __post_init__(self) -> None:
        finite_values = (
            self.box_x,
            self.goal_x,
            *self.half_size,
            self.box_mass,
            *self.box_friction,
            self.box_slide_damping,
            self.goal_half_size,
            self.hand_half_width,
            self.hand_target_world_z,
            *self.palm_normal,
            self.navigation_speed,
            self.push_speed,
            self.standoff,
            self.contact_dwell,
            self.contact_loss,
            self.timeout,
            self.stall_seconds,
            self.progress_distance,
            *self.weld_solref,
            self.virtual_force_magnitude,
            self.virtual_force_max,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("Box-push settings must be finite")
        if self.box_x <= 1.0 or self.goal_x <= self.box_x:
            raise ValueError(
                "Box must start ahead of the robot, with the goal beyond it"
            )
        if (
            any(size <= 0.0 for size in self.half_size)
            or self.box_mass <= 0.0
            or self.box_slide_damping <= 0.0
        ):
            raise ValueError("Box dimensions, mass, and damping must be positive")
        if self.goal_half_size <= 0.0 or self.hand_half_width < 0.0:
            raise ValueError("Goal and hand-spacing settings are invalid")
        if not 0.0 <= self.hand_target_world_z <= self.box_top_z:
            raise ValueError("Hand target must lie between the box floor and top")
        if math.sqrt(sum(value * value for value in self.palm_normal)) <= 1e-6:
            raise ValueError("Palm normal must be non-zero")
        if not all(
            value > 0.0
            for value in (
                self.navigation_speed,
                self.push_speed,
                self.standoff,
                self.contact_dwell,
                self.contact_loss,
                self.timeout,
                self.stall_seconds,
                self.progress_distance,
                self.weld_solref[0],
            )
        ):
            raise ValueError("Box-push pacing settings must be positive")
        if self.contact_windows < 1 or self.reacquisitions < 0:
            raise ValueError("Box-push window settings are invalid")
        if not all(
            prompt.strip()
            for prompt in (
                self.approach_prompt,
                self.contact_prompt,
                self.push_prompt,
                self.settle_prompt,
            )
        ):
            raise ValueError("Box-push prompts must be non-empty")


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
    if value not in {"false", "true"}:
        raise ValueError(f"{name} must be 'false' or 'true'")
    return value == "true"
