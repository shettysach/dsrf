"""Simulation-facing controller contract and implementations."""

from __future__ import annotations

from typing import Protocol

import torch

from shared.messages import MotionChunk
from sim.env import RobotState


class Controller(Protocol):
    """Turns motion chunks and robot state into simulation actions."""

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None: ...

    def act(self, state: RobotState) -> tuple[torch.Tensor, bool]: ...
