from __future__ import annotations

from pathlib import Path

import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
)

from shared.g1 import (
    DEFAULT_JOINT_POS_MJLAB,
    MJLAB_FROM_SONIC,
    SONIC_FROM_MJLAB,
)
from shared.messages import MotionChunk
from tracker import RobotState
from tracker.reference import MotionReference
from tracker.sonic.observations import ObservationLayout
from tracker.sonic.onnx_model import OnnxModel

HISTORY_FRAMES = 10


class SonicTracker:
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

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None:
        if len(chunk.qpos) < 2:
            raise ValueError("SONIC requires at least two reference frames")
        self.reference.load(chunk, state.root_pos_w, state.root_quat_w)

    def act(self, state: RobotState) -> tuple[torch.Tensor, bool]:
        joint_position = (state.joint_pos - self._default_joint_pos).index_select(
            0, self._sonic_from_mjlab
        )
        joint_velocity = state.joint_vel.index_select(0, self._sonic_from_mjlab)
        positions, velocities, reference_quats = self.reference.window(
            count=HISTORY_FRAMES, step=self.layout.g1_step
        )
        positions = positions.index_select(1, self._sonic_from_mjlab)
        velocities = velocities.index_select(1, self._sonic_from_mjlab)
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
