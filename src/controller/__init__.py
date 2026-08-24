"""Controller contract and implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from shared.g1 import G1_JOINT_COUNT
from shared.messages import MotionChunk


@dataclass(frozen=True)
class ExternalWrench:
    """World-frame wrench applied at a named body's center of mass."""

    body: str
    force_w: torch.Tensor
    torque_w: torch.Tensor

    def __post_init__(self) -> None:
        if not self.body.strip():
            raise ValueError("External wrench body must not be empty")
        for name, value in (("force_w", self.force_w), ("torque_w", self.torque_w)):
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {value.shape}")


@dataclass(frozen=True)
class ControlOutput:
    """Physical command produced by a controller for one robot."""

    joint_target: torch.Tensor
    completed: bool = False
    external_wrenches: tuple[ExternalWrench, ...] = ()
    joint_velocity_target: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.joint_target.shape != (G1_JOINT_COUNT,):
            raise ValueError(
                f"joint_target must have shape ({G1_JOINT_COUNT},), "
                f"got {self.joint_target.shape}"
            )
        if (
            self.joint_velocity_target is not None
            and self.joint_velocity_target.shape != (G1_JOINT_COUNT,)
        ):
            raise ValueError(
                f"joint_velocity_target must have shape ({G1_JOINT_COUNT},), "
                f"got {self.joint_velocity_target.shape}"
            )


@dataclass(frozen=True)
class RobotState:
    """Robot state supplied by a simulator to a controller."""

    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_states: dict[str, BodyState] = field(default_factory=dict)


@dataclass(frozen=True)
class BodyState:
    """World-frame state for a named rigid body."""

    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


class Controller(Protocol):
    """Turns motion chunks and robot state into simulation actions."""

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None: ...

    def act(self, state: RobotState) -> ControlOutput: ...
