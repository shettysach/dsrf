from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_box_minus

from controller import ControlOutput, ExternalWrench, RobotState
from controller.reference import MotionReference, ReferenceFrame
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
        force_w, torque_w = self._root_wrench(target, state)
        return ControlOutput(
            joint_target=target.joint_pos,
            joint_velocity_target=target.joint_vel,
            external_wrenches=(
                ExternalWrench("pelvis", force_w=force_w, torque_w=torque_w),
            ),
            completed=completed,
        )

    def _root_wrench(
        self, target: ReferenceFrame, state: RobotState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the world-frame spring-damper wrench for the pelvis."""
        force_w = (
            self.config.root_pos_kp * (target.root_pos_w - state.root_pos_w)
            + self.config.root_pos_kd
            * (target.root_lin_vel_w - state.root_lin_vel_w)
        )
        orientation_error = quat_box_minus(
            target.root_quat_w[None], state.root_quat_w[None]
        )[0]
        torque_w = (
            self.config.root_rot_kp * orientation_error
            + self.config.root_rot_kd
            * (target.root_ang_vel_w - state.root_ang_vel_w)
        )
        return (
            _clamp_norm(force_w, self.config.max_force),
            _clamp_norm(torque_w, self.config.max_torque),
        )


def _clamp_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    limit = torch.as_tensor(maximum, dtype=vector.dtype, device=vector.device)
    scale = torch.clamp(limit / norm.clamp_min(torch.finfo(vector.dtype).eps), max=1.0)
    return vector * scale
