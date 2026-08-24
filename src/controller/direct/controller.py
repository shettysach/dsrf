from __future__ import annotations

import torch

from controller import ControlOutput, RobotState
from controller.reference import MotionReference
from shared.config import DirectConfig
from shared.messages import MotionChunk


class DirectController:
    """Pure motion-reference tracking through MJLab's built-in joint PD."""

    def __init__(
        self,
        config: DirectConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        del config
        self.reference = MotionReference(device)

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None:
        self.reference.load(chunk, state.root_pos_w, state.root_quat_w)

    def act(self, state: RobotState) -> ControlOutput:
        del state
        target = self.reference.current()
        completed = self.reference.advance()
        return ControlOutput(
            joint_target=target.joint_pos,
            joint_velocity_target=target.joint_vel,
            completed=completed,
        )
