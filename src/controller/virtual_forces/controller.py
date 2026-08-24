from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_box_minus

from controller import ControlOutput, ExternalWrench, RobotState
from controller.reference import MotionReference
from shared.config import VirtualForcesConfig
from shared.messages import MotionChunk


class VirtualForcesController:
    """Track physical joint references, optionally assisting the floating base."""

    def __init__(
        self,
        config: VirtualForcesConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.reference = MotionReference(device)

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None:
        self.reference.load(chunk, state.root_pos_w, state.root_quat_w)

    def act(self, state: RobotState) -> ControlOutput:
        target = self.reference.current()
        wrenches: tuple[ExternalWrench, ...] = ()
        if self.config.assistance_enabled:
            position_error = _with_deadband(
                target.root_pos_w - state.root_pos_w,
                self.config.position_deadband,
            )
            orientation_error = _with_deadband(
                quat_box_minus(
                    target.root_quat_w.unsqueeze(0),
                    state.root_quat_w.unsqueeze(0),
                ).squeeze(0),
                self.config.orientation_deadband,
            )
            force = _limit_norm(
                self.config.position_kp * position_error
                + self.config.position_kd
                * (target.root_lin_vel_w - state.root_lin_vel_w),
                self.config.force_limit,
            )
            torque = _limit_norm(
                self.config.orientation_kp * orientation_error
                + self.config.orientation_kd
                * (target.root_ang_vel_w - state.root_ang_vel_w),
                self.config.torque_limit,
            )
            wrenches = (
                ExternalWrench(
                    body=self.config.target_body,
                    force_w=force,
                    torque_w=torque,
                ),
            )

        completed = self.reference.advance()
        return ControlOutput(
            joint_target=target.joint_pos,
            completed=completed,
            external_wrenches=wrenches,
            joint_velocity_target=target.joint_vel,
        )


def _with_deadband(value: torch.Tensor, deadband: float) -> torch.Tensor:
    return torch.where(
        torch.linalg.vector_norm(value) <= deadband,
        torch.zeros_like(value),
        value,
    )


def _limit_norm(value: torch.Tensor, limit: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value)
    scale = torch.clamp(limit / torch.clamp_min(norm, 1.0e-12), max=1.0)
    return value * scale
