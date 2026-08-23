"""Controller contract and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from shared.messages import MotionChunk


@dataclass(frozen=True)
class RobotState:
    """Robot state supplied by a simulator to a controller."""

    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class Controller(Protocol):
    """Turns motion chunks and robot state into simulation actions."""

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None: ...

    def act(self, state: RobotState) -> tuple[torch.Tensor, bool]: ...
