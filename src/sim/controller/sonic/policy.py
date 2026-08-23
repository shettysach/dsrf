from __future__ import annotations

from pathlib import Path

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

from shared.g1 import (
    DEFAULT_JOINT_POS_MJLAB,
    MJLAB_FROM_SONIC,
    SONIC_FROM_MJLAB,
    standing_qpos,
)
from shared.messages import SONIC_FPS, MotionChunk
from sim.controller.sonic.onnx_model import OnnxModel
from sim.env import RobotState
from sim.observations import ObservationLayout

HISTORY_FRAMES = 10


class MotionReference:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._qpos = torch.as_tensor(
            standing_qpos()[None], dtype=torch.float32, device=device
        )
        self._joint_vel = torch.zeros((1, 29), dtype=torch.float32, device=device)
        self._heading_delta = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device
        )
        self._robot_origin_w = torch.zeros(3, dtype=torch.float32, device=device)
        self._reference_origin = torch.zeros(3, dtype=torch.float32, device=device)
        self._sonic_from_mjlab = torch.as_tensor(
            SONIC_FROM_MJLAB, dtype=torch.long, device=device
        )
        self._future_offsets = torch.arange(
            HISTORY_FRAMES, dtype=torch.long, device=device
        )
        self._frame = 0
        self._active = False

    def load(
        self,
        chunk: MotionChunk,
        robot_pos_w: torch.Tensor,
        robot_quat_w: torch.Tensor,
    ) -> None:
        self._qpos = torch.tensor(
            chunk.qpos, dtype=torch.float32, device=self.device
        ).contiguous()
        if len(self._qpos) < 2:
            raise ValueError("SONIC requires at least two reference frames")
        natural_positions = self._qpos[:, 7:]
        velocities = torch.empty_like(natural_positions)
        velocities[:-1] = torch.diff(natural_positions, dim=0) * SONIC_FPS
        velocities[-1] = velocities[-2]
        self._joint_vel = velocities

        reference_quat = self._qpos[0, 3:7]
        self._heading_delta = quat_mul(
            yaw_quat(robot_quat_w), quat_conjugate(yaw_quat(reference_quat))
        )
        self._robot_origin_w = robot_pos_w.clone()
        self._reference_origin = self._qpos[0, :3].clone()
        self._frame = 0
        self._active = True

    def visualization_pose(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return the active reference pose aligned to the command start pose."""
        if not self._active:
            return None

        qpos = self._qpos[self._frame]
        root_pos_w = self._robot_origin_w + quat_apply(
            self._heading_delta, qpos[:3] - self._reference_origin
        )
        root_pos_w[2] = qpos[2]  # NOTE: No Z rebasing, only X Y
        root_quat_w = quat_mul(self._heading_delta, qpos[3:7])
        return root_pos_w, root_quat_w, qpos[7:]

    def window(self, *, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.clamp(
            self._future_offsets * step + self._frame,
            max=len(self._qpos) - 1,
        )
        qpos = self._qpos.index_select(0, indices)
        natural_positions = qpos[:, 7:]
        natural_velocities = self._joint_vel.index_select(0, indices)
        positions = natural_positions.index_select(1, self._sonic_from_mjlab)
        velocities = natural_velocities.index_select(1, self._sonic_from_mjlab)
        reference_quats = qpos[:, 3:7]
        aligned_quats = quat_mul(
            self._heading_delta.expand_as(reference_quats), reference_quats
        )
        return positions, velocities, aligned_quats

    def advance(self) -> bool:
        if not self._active:
            return False
        if self._frame < len(self._qpos) - 1:
            self._frame += 1
            return False
        self._active = False
        return True


class SonicPolicy:
    """SONIC inference with CPU execution or zero-copy CUDA I/O binding."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        device: str = "cpu",
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.layout = ObservationLayout.load(bundle_dir / "observation_config.yaml")
        self.encoder = OnnxModel(
            bundle_dir / "model_encoder.onnx",
            input_shape=(1, self.layout.encoder_input_dimension),
            output_shape=(1, self.layout.encoder_dimension),
            device=self.device,
            cuda_stream=cuda_stream,
        )
        self.decoder = OnnxModel(
            bundle_dir / "model_decoder.onnx",
            input_shape=(1, self.layout.policy_input_dimension),
            output_shape=(1, 29),
            device=self.device,
            cuda_stream=cuda_stream,
        )
        self._default_joint_pos = torch.as_tensor(
            DEFAULT_JOINT_POS_MJLAB, dtype=torch.float32, device=self.device
        )
        self._sonic_from_mjlab = torch.as_tensor(
            SONIC_FROM_MJLAB, dtype=torch.long, device=self.device
        )
        self._mjlab_from_sonic = torch.as_tensor(
            MJLAB_FROM_SONIC, dtype=torch.long, device=self.device
        )
        self._g1_encoder_mode = torch.zeros(4, dtype=torch.float32, device=self.device)
        self.reference = MotionReference(self.device)
        self._last_action = torch.zeros(29, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.reference = MotionReference(self.device)
        self._last_action.zero_()
        self.encoder.input.zero_()
        self.decoder.input.zero_()

    def load_motion(
        self,
        chunk: MotionChunk,
        robot_pos_w: torch.Tensor,
        robot_quat_w: torch.Tensor,
    ) -> None:
        self.reference.load(chunk, robot_pos_w, robot_quat_w)

    def infer(self, state: RobotState) -> tuple[torch.Tensor, bool]:
        joint_position = (state.joint_pos - self._default_joint_pos).index_select(
            0, self._sonic_from_mjlab
        )
        joint_velocity = state.joint_vel.index_select(0, self._sonic_from_mjlab)
        positions, velocities, reference_quats = self.reference.window(
            step=self.layout.g1_step
        )
        relative_quats = quat_mul(
            quat_conjugate(state.root_quat_w).expand_as(reference_quats),
            reference_quats,
        )
        orientation_6d = matrix_from_quat(relative_quats)[..., :2]
        suffix = f"10frame_step{self.layout.g1_step}"
        self._copy_encoder("encoder_mode_4", self._g1_encoder_mode)
        self._copy_encoder(f"motion_joint_positions_{suffix}", positions)
        self._copy_encoder(f"motion_joint_velocities_{suffix}", velocities)
        self._copy_encoder(f"motion_anchor_orientation_{suffix}", orientation_6d)
        token = self.encoder.run()

        self._copy_policy("token_state", token)
        self._append_history(
            "his_base_angular_velocity_10frame_step1",
            state.root_ang_vel_b,
        )
        self._append_history("his_body_joint_positions_10frame_step1", joint_position)
        self._append_history("his_body_joint_velocities_10frame_step1", joint_velocity)
        self._append_history("his_last_actions_10frame_step1", self._last_action)
        self._append_history(
            "his_gravity_dir_10frame_step1",
            state.projected_gravity_b,
        )
        action_sonic = self.decoder.run().reshape(29)
        if self.device.type == "cpu" and not bool(torch.isfinite(action_sonic).all()):
            raise RuntimeError("SONIC decoder returned NaN or infinite actions")
        self._last_action.copy_(action_sonic)
        completed = self.reference.advance()
        action_mjlab = action_sonic.index_select(0, self._mjlab_from_sonic)
        return action_mjlab.unsqueeze(0), completed

    def _copy_encoder(self, name: str, value: torch.Tensor) -> None:
        self.encoder.input[0, self.layout.encoder_slices[name]].copy_(value.reshape(-1))

    def _copy_policy(self, name: str, value: torch.Tensor) -> None:
        self.decoder.input[0, self.layout.policy_slices[name]].copy_(value.reshape(-1))

    def _append_history(self, name: str, value: torch.Tensor) -> None:
        history = self.decoder.input[0, self.layout.policy_slices[name]].view(
            HISTORY_FRAMES, -1
        )
        history[:-1].copy_(history[1:].clone())
        history[-1].copy_(value.reshape(-1))
