from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import torch
from mjlab.utils.lab_api.math import quat_box_minus

from controller import ControlOutput, ExternalWrench, RobotState
from controller.reference import MotionReference, ReferenceFrame
from shared.config import DirectConfig
from shared.messages import REFERENCE_HZ, MotionChunk


@dataclass(frozen=True)
class RootTrackingDiagnostic:
    position_error_w: torch.Tensor
    velocity_error_w: torch.Tensor
    orientation_error_w: torch.Tensor
    angular_velocity_error_w: torch.Tensor
    force_w: torch.Tensor
    torque_w: torch.Tensor
    force_clipped: bool
    torque_clipped: bool


@dataclass(frozen=True)
class BodyReference:
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


VIRTUAL_SPRING_BODIES = (
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


class DirectController:
    """Send motion-reference joint targets to MJLab's built-in PD actuators."""

    def __init__(
        self,
        config: DirectConfig,
        *,
        robot_mass: float,
        gravity_magnitude: float,
        mj_model: object | None = None,
        robot_indexing: object | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if robot_mass <= 0.0 or gravity_magnitude <= 0.0:
            raise ValueError("robot_mass and gravity_magnitude must be positive")
        self.config = config
        self.reference = MotionReference(device)
        self._gravity_force = (
            config.root_gravity_support * robot_mass * gravity_magnitude
        )
        self._reference_kinematics = (
            _ReferenceKinematics(mj_model, robot_indexing, self.reference.device)
            if mj_model is not None and robot_indexing is not None
            else None
        )
        self._body_references: dict[str, tuple[BodyReference, ...]] = {}
        self._wrench_logger = _WrenchCsvLogger(
            config.wrench_log_path, config.max_force, config.max_torque
        )
        self._tracking_frame = 0
        self._observation_id = 0

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None:
        self.reference.load(chunk, state.root_pos_w, state.root_quat_w)
        if self._reference_kinematics is not None:
            self._body_references = self._reference_kinematics.trajectory(self.reference)
        self._tracking_frame = 0
        self._observation_id = chunk.observation_id

    def act(self, state: RobotState) -> ControlOutput:
        target = self.reference.current()
        completed = self.reference.advance()
        diagnostic = self._root_tracking_diagnostic(target, state)
        body_wrenches = self._body_wrenches(self.reference.frame_index, state)
        wrenches = (
            ExternalWrench(
                "pelvis",
                force_w=diagnostic.force_w,
                torque_w=diagnostic.torque_w,
            ),
            *body_wrenches,
        )
        self._wrench_logger.write(
            observation_id=self._observation_id,
            frame=self._tracking_frame,
            target=target,
            state=state,
            diagnostic=diagnostic,
            wrenches=wrenches,
        )
        self._tracking_frame += 1
        return ControlOutput(
            joint_target=target.joint_pos,
            joint_velocity_target=target.joint_vel,
            external_wrenches=wrenches,
            completed=completed,
        )

    def _root_wrench(
        self, target: ReferenceFrame, state: RobotState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the world-frame spring-damper wrench for the pelvis."""
        diagnostic = self._root_tracking_diagnostic(target, state)
        return diagnostic.force_w, diagnostic.torque_w

    def _root_tracking_diagnostic(
        self, target: ReferenceFrame, state: RobotState
    ) -> RootTrackingDiagnostic:
        position_error = target.root_pos_w - state.root_pos_w
        velocity_error = target.root_lin_vel_w - state.root_lin_vel_w
        force_w = torch.stack(
            (
                torch.zeros((), dtype=position_error.dtype, device=position_error.device),
                torch.zeros((), dtype=position_error.dtype, device=position_error.device),
                position_error.new_tensor(self._gravity_force)
                + self.config.root_z_kp * position_error[2]
                + self.config.root_z_kd * velocity_error[2],
            )
        )
        orientation_error = quat_box_minus(
            target.root_quat_w[None], state.root_quat_w[None]
        )[0]
        angular_velocity_error = target.root_ang_vel_w - state.root_ang_vel_w
        torque_w = torch.stack(
            (
                self.config.root_rp_kp * orientation_error[0]
                + self.config.root_rp_kd * angular_velocity_error[0],
                self.config.root_rp_kp * orientation_error[1]
                + self.config.root_rp_kd * angular_velocity_error[1],
                torch.zeros(
                    (), dtype=orientation_error.dtype, device=orientation_error.device
                ),
            )
        )
        force_norm = torch.linalg.vector_norm(force_w)
        torque_norm = torch.linalg.vector_norm(torque_w)
        return RootTrackingDiagnostic(
            position_error_w=position_error,
            velocity_error_w=velocity_error,
            orientation_error_w=orientation_error,
            angular_velocity_error_w=angular_velocity_error,
            force_w=_clamp_norm(force_w, self.config.max_force),
            torque_w=_clamp_norm(torque_w, self.config.max_torque),
            force_clipped=bool(force_norm > self.config.max_force),
            torque_clipped=bool(torque_norm > self.config.max_torque),
        )

    def _body_wrenches(
        self, frame: int, state: RobotState
    ) -> tuple[ExternalWrench, ...]:
        wrenches: list[ExternalWrench] = []
        for body in VIRTUAL_SPRING_BODIES[1:]:
            if body not in state.body_states or body not in self._body_references:
                continue
            target = self._body_references[body][frame]
            current = state.body_states[body]
            if body.endswith("ankle_roll_link"):
                force = _clamp_norm(
                    self.config.foot_pos_kp * (target.pos_w - current.pos_w)
                    + self.config.foot_pos_kd * (target.lin_vel_w - current.lin_vel_w),
                    self.config.max_force,
                )
                torque = torch.zeros_like(force)
            else:
                force = torch.zeros_like(current.pos_w)
                rotation_error = quat_box_minus(target.quat_w[None], current.quat_w[None])[0]
                angular_error = target.ang_vel_w - current.ang_vel_w
                torque = torch.stack(
                    (
                        self.config.torso_rp_kp * rotation_error[0]
                        + self.config.torso_rp_kd * angular_error[0],
                        self.config.torso_rp_kp * rotation_error[1]
                        + self.config.torso_rp_kd * angular_error[1],
                        torch.zeros((), dtype=force.dtype, device=force.device),
                    )
                )
                torque = _clamp_norm(torque, self.config.max_torque)
            wrenches.append(ExternalWrench(body, force, torque))
        return tuple(wrenches)


def _clamp_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    limit = torch.as_tensor(maximum, dtype=vector.dtype, device=vector.device)
    scale = torch.clamp(limit / norm.clamp_min(torch.finfo(vector.dtype).eps), max=1.0)
    return vector * scale


class _ReferenceKinematics:
    """CPU MuJoCo forward kinematics for the selected virtual-spring bodies."""

    def __init__(
        self, model: object, indexing: object, device: torch.device
    ) -> None:
        self._model = model
        self._data = mujoco.MjData(model)
        self._device = device
        self._free_joint_q_adr = indexing.free_joint_q_adr.detach().cpu().numpy()
        self._joint_q_adr = indexing.joint_q_adr.detach().cpu().numpy()
        self._body_ids = {
            name: _model_body_id(model, name) for name in VIRTUAL_SPRING_BODIES
        }

    def trajectory(
        self, reference: MotionReference
    ) -> dict[str, tuple[BodyReference, ...]]:
        root_pos, root_quat, joint_pos = reference.trajectory()
        frames: dict[str, list[BodyReference]] = {
            name: [] for name in VIRTUAL_SPRING_BODIES
        }
        for position, quaternion, joints in zip(
            root_pos, root_quat, joint_pos, strict=True
        ):
            self._data.qpos.fill(0.0)
            self._data.qpos[self._free_joint_q_adr[:3]] = position.detach().cpu().numpy()
            self._data.qpos[self._free_joint_q_adr[3:7]] = quaternion.detach().cpu().numpy()
            self._data.qpos[self._joint_q_adr] = joints.detach().cpu().numpy()
            mujoco.mj_forward(self._model, self._data)
            for name, body_id in self._body_ids.items():
                frames[name].append(
                    BodyReference(
                        pos_w=torch.as_tensor(
                            self._data.xpos[body_id].copy(),
                            dtype=torch.float32,
                            device=self._device,
                        ),
                        quat_w=torch.as_tensor(
                            self._data.xquat[body_id].copy(),
                            dtype=torch.float32,
                            device=self._device,
                        ),
                        lin_vel_w=torch.zeros(3, device=self._device),
                        ang_vel_w=torch.zeros(3, device=self._device),
                    )
                )
        return {
            name: _with_velocities(body_frames)
            for name, body_frames in frames.items()
        }


def _model_body_id(model: Any, name: str) -> int:
    try:
        return int(model.body(name).id)
    except KeyError:
        return int(model.body(f"robot/{name}").id)


def _with_velocities(frames: list[BodyReference]) -> tuple[BodyReference, ...]:
    positions = torch.stack([frame.pos_w for frame in frames])
    quaternions = torch.stack([frame.quat_w for frame in frames])
    linear_velocity = _finite_difference(positions)
    angular_velocity = torch.zeros_like(linear_velocity)
    if len(frames) > 1:
        angular_velocity[:-1] = (
            quat_box_minus(quaternions[1:], quaternions[:-1]) * REFERENCE_HZ
        )
        angular_velocity[-1] = angular_velocity[-2]
    return tuple(
        BodyReference(
            pos_w=frame.pos_w,
            quat_w=frame.quat_w,
            lin_vel_w=linear_velocity[index],
            ang_vel_w=angular_velocity[index],
        )
        for index, frame in enumerate(frames)
    )


def _finite_difference(values: torch.Tensor) -> torch.Tensor:
    velocity = torch.zeros_like(values)
    if len(values) > 1:
        velocity[:-1] = torch.diff(values, dim=0) * REFERENCE_HZ
        velocity[-1] = velocity[-2]
    return velocity


class _WrenchCsvLogger:
    _FIELDNAMES = (
        "time_s", "observation_id", "frame",
        "root_x", "root_y", "root_z",
        "ref_root_x", "ref_root_y", "ref_root_z",
        "err_x", "err_y", "err_z",
        "vel_x", "vel_y", "vel_z",
        "ref_vel_x", "ref_vel_y", "ref_vel_z",
        "rot_err_x", "rot_err_y", "rot_err_z",
        "ang_vel_err_x", "ang_vel_err_y", "ang_vel_err_z",
        "force_x", "force_y", "force_z",
        "torque_x", "torque_y", "torque_z",
        "force_norm", "torque_norm", "force_clipped", "torque_clipped",
    )
    _BODY_WRENCH_FIELDS = tuple(
        f"{body}_{quantity}_{component}"
        for body in VIRTUAL_SPRING_BODIES
        for quantity in ("force", "torque")
        for component in "xyz"
    ) + tuple(
        f"{body}_{quantity}"
        for body in VIRTUAL_SPRING_BODIES
        for quantity in ("force_norm", "torque_norm", "force_clipped", "torque_clipped")
    )

    def __init__(self, path: str, max_force: float, max_torque: float) -> None:
        self._file = None
        self._writer: csv.DictWriter[str] | None = None
        self._max_force = max_force
        self._max_torque = max_torque
        if not path:
            return
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = output_path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._FIELDNAMES + self._BODY_WRENCH_FIELDS
        )
        self._writer.writeheader()

    def write(
        self,
        *,
        observation_id: int,
        frame: int,
        target: ReferenceFrame,
        state: RobotState,
        diagnostic: RootTrackingDiagnostic,
        wrenches: tuple[ExternalWrench, ...],
    ) -> None:
        if self._writer is None or self._file is None:
            return
        row: dict[str, float | int | bool] = {
            "time_s": frame / REFERENCE_HZ,
            "observation_id": observation_id,
            "frame": frame,
            "force_norm": _scalar(torch.linalg.vector_norm(diagnostic.force_w)),
            "torque_norm": _scalar(torch.linalg.vector_norm(diagnostic.torque_w)),
            "force_clipped": diagnostic.force_clipped,
            "torque_clipped": diagnostic.torque_clipped,
        }
        row.update(_components("root", state.root_pos_w))
        row.update(_components("ref_root", target.root_pos_w))
        row.update(_components("err", diagnostic.position_error_w))
        row.update(_components("vel", state.root_lin_vel_w))
        row.update(_components("ref_vel", target.root_lin_vel_w))
        row.update(_components("rot_err", diagnostic.orientation_error_w))
        row.update(_components("ang_vel_err", diagnostic.angular_velocity_error_w))
        row.update(_components("force", diagnostic.force_w))
        row.update(_components("torque", diagnostic.torque_w))
        for wrench in wrenches:
            row.update(_components(f"{wrench.body}_force", wrench.force_w))
            row.update(_components(f"{wrench.body}_torque", wrench.torque_w))
            force_norm = _scalar(torch.linalg.vector_norm(wrench.force_w))
            torque_norm = _scalar(torch.linalg.vector_norm(wrench.torque_w))
            row[f"{wrench.body}_force_norm"] = force_norm
            row[f"{wrench.body}_torque_norm"] = torque_norm
            row[f"{wrench.body}_force_clipped"] = force_norm >= 0.999 * self._max_force
            row[f"{wrench.body}_torque_clipped"] = torque_norm >= 0.999 * self._max_torque
        self._writer.writerow(row)
        self._file.flush()


def _components(prefix: str, values: torch.Tensor) -> dict[str, float]:
    return {
        f"{prefix}_{axis}": _scalar(value)
        for axis, value in zip("xyz", values, strict=True)
    }


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())
