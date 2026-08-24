from __future__ import annotations

import torch

from controller import ControlOutput, RobotState, RootTarget
from controller.reference import MotionReference
from shared.config import DirectConfig
from shared.messages import MotionChunk


class DirectController:
    """Send motion-reference joint targets to MJLab's built-in PD actuators."""

    def __init__(
        self,
        config: DirectConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.reference = MotionReference(device)

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None:
        self.reference.load(chunk, state.root_pos_w, state.root_quat_w)

    def act(self, state: RobotState) -> ControlOutput:
        target = self.reference.current()
        completed = self.reference.advance()
        root_target = (
            RootTarget(
                pos_w=target.root_pos_w,
                quat_w=target.root_quat_w,
                lin_vel_w=target.root_lin_vel_w,
                ang_vel_w=target.root_ang_vel_w,
            )
            if self.config.pin_root
            else None
        )
        return ControlOutput(
            joint_target=target.joint_pos,
            joint_velocity_target=target.joint_vel,
            root_target=root_target,
            completed=completed,
        )
