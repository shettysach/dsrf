"""Reference-acceleration WBC that executes through MJLab's PD actuators."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_box_minus

from controller import ControlOutput, DynamicsSnapshot, RobotState
from controller.reference import MotionReference
from controller.wbc.qp import solve_inverse_dynamics
from shared.config import WbcConfig
from shared.messages import MotionChunk


class WbcController:
    """Contact-consistent correction of ARDY's normal joint PD command."""

    def __init__(self, config: WbcConfig, *, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.reference = MotionReference(device)
        self.last_failure: str | None = None

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None:
        self.reference.load(chunk, state.root_pos_w, state.root_quat_w)

    def act(
        self, state: RobotState, dynamics: DynamicsSnapshot | None = None
    ) -> ControlOutput:
        if dynamics is None:
            raise ValueError("WBC requires a dynamics snapshot")
        target = self.reference.current()
        completed = self.reference.advance()
        fallback = ControlOutput(
            joint_target=target.joint_pos,
            joint_velocity_target=target.joint_vel,
            completed=completed,
        )
        if not dynamics.contacts:
            return fallback
        try:
            qacc_des = self._desired_acceleration(state, target, dynamics)
            joint_velocity_error = target.joint_vel - state.joint_vel
            torque_pd = dynamics.joint_stiffness * (target.joint_pos - state.joint_pos)
            torque_pd = torque_pd + dynamics.joint_damping * joint_velocity_error
            solved = solve_inverse_dynamics(
                dynamics,
                qacc_des,
                torque_pd,
                acceleration_weight=self.config.acceleration_weight,
                pd_weight=self.config.pd_weight,
                force_weight=self.config.force_weight,
            )
            if not torch.isfinite(solved.torque).all():
                raise RuntimeError("non-finite KKT solution")
            command = state.joint_pos + (
                solved.torque - dynamics.joint_damping * joint_velocity_error
            ) / dynamics.joint_stiffness
            self.last_failure = None
            return ControlOutput(
                joint_target=command,
                joint_velocity_target=target.joint_vel,
                completed=completed,
            )
        except (RuntimeError, torch.linalg.LinAlgError) as exc:
            self.last_failure = str(exc)
            return fallback

    def _desired_acceleration(self, state: RobotState, target, dynamics: DynamicsSnapshot) -> torch.Tensor:
        position_error = target.root_pos_w - state.root_pos_w
        rotation_error = quat_box_minus(target.root_quat_w, state.root_quat_w)
        root_linear = target.root_lin_acc_w + self.config.root_translation_kp * position_error
        root_linear = root_linear + self.config.root_translation_kd * (
            target.root_lin_vel_w - state.root_lin_vel_w
        )
        root_angular = target.root_ang_acc_w + self.config.root_rotation_kp * rotation_error
        root_angular = root_angular + self.config.root_rotation_kd * (
            target.root_ang_vel_w - state.root_ang_vel_w
        )
        joint = target.joint_acc + self.config.joints_kp * (target.joint_pos - state.joint_pos)
        joint = joint + self.config.joints_kd * (target.joint_vel - state.joint_vel)
        qacc = torch.cat((root_linear, root_angular, joint))
        if qacc.shape != dynamics.qvel.shape:
            raise ValueError("Dynamics snapshot generalized velocity ordering is invalid")
        return qacc
