from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

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
        self._wrench_logger = _WrenchCsvLogger(config.wrench_log_path)
        self._tracking_frame = 0
        self._observation_id = 0

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None:
        self.reference.load(chunk, state.root_pos_w, state.root_quat_w)
        self._tracking_frame = 0
        self._observation_id = chunk.observation_id

    def act(self, state: RobotState) -> ControlOutput:
        target = self.reference.current()
        completed = self.reference.advance()
        diagnostic = self._root_tracking_diagnostic(target, state)
        self._wrench_logger.write(
            observation_id=self._observation_id,
            frame=self._tracking_frame,
            target=target,
            state=state,
            diagnostic=diagnostic,
        )
        self._tracking_frame += 1
        return ControlOutput(
            joint_target=target.joint_pos,
            joint_velocity_target=target.joint_vel,
            external_wrenches=(
                ExternalWrench(
                    "pelvis",
                    force_w=diagnostic.force_w,
                    torque_w=diagnostic.torque_w,
                ),
            ),
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
        position_kp = position_error.new_tensor(
            (self.config.root_xy_kp, self.config.root_xy_kp, self.config.root_z_kp)
        )
        position_kd = velocity_error.new_tensor(
            (self.config.root_xy_kd, self.config.root_xy_kd, self.config.root_z_kd)
        )
        force_w = (
            position_kp * position_error + position_kd * velocity_error
        )
        orientation_error = quat_box_minus(
            target.root_quat_w[None], state.root_quat_w[None]
        )[0]
        angular_velocity_error = target.root_ang_vel_w - state.root_ang_vel_w
        rotation_kp = orientation_error.new_tensor(
            (self.config.root_rp_kp, self.config.root_rp_kp, self.config.root_yaw_kp)
        )
        rotation_kd = angular_velocity_error.new_tensor(
            (self.config.root_rp_kd, self.config.root_rp_kd, self.config.root_yaw_kd)
        )
        torque_w = (
            rotation_kp * orientation_error + rotation_kd * angular_velocity_error
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


def _clamp_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    limit = torch.as_tensor(maximum, dtype=vector.dtype, device=vector.device)
    scale = torch.clamp(limit / norm.clamp_min(torch.finfo(vector.dtype).eps), max=1.0)
    return vector * scale


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

    def __init__(self, path: str) -> None:
        self._file = None
        self._writer: csv.DictWriter[str] | None = None
        if not path:
            return
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = output_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDNAMES)
        self._writer.writeheader()

    def write(
        self,
        *,
        observation_id: int,
        frame: int,
        target: ReferenceFrame,
        state: RobotState,
        diagnostic: RootTrackingDiagnostic,
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
        self._writer.writerow(row)
        self._file.flush()


def _components(prefix: str, values: torch.Tensor) -> dict[str, float]:
    return {
        f"{prefix}_{axis}": _scalar(value)
        for axis, value in zip("xyz", values, strict=True)
    }


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())
