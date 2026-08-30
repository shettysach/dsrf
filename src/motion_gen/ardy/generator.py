from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import load_model
from ardy.motion_rep.tools import length_to_mask

from motion_gen.ardy.constraints import (
    _JOINT_NAMES,
    build_constraints,
    end_effector_target_positions,
)
from motion_gen.ardy.encoder import prepare_conditioning
from motion_gen.ardy.history import build_initial_history, qpos_to_ardy_inputs
from shared.g1 import standing_qpos
from shared.messages import EndEffectorTarget

_LOGGER = logging.getLogger(__name__)


class Ardy:
    """Text-conditioned ARDY motion generator for Unitree G1."""

    fps = 25
    duration_s = 5

    def __init__(
        self,
        checkpoints_dir: Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = load_model(
            "g1",
            device=str(self.device),
            checkpoints_dir=str(checkpoints_dir),
        )
        model_fps = float(self.model.motion_rep.fps)
        if model_fps != self.fps:
            raise ValueError(f"Expected ARDY G1 at {self.fps} FPS, got {model_fps}")

        self.converter = MujocoQposConverter(self.model.skeleton)
        self.history_frames = int(self.model.num_frames_per_token)
        standing_history = np.repeat(standing_qpos()[None], self.history_frames, axis=0)
        self.initial_history = build_initial_history(
            standing_history,
            self.converter,
            self.model.motion_rep,
            device=self.device,
        )
        _, root_positions = qpos_to_ardy_inputs(
            standing_history,
            self.converter,
            device=self.device,
        )
        self.root_history = root_positions[0, -2:].detach().clone()
        self.root_heading = torch.tensor(0.0)

    def generate(
        self,
        embedding: torch.Tensor,
        target_xys: tuple[tuple[float, float], ...],
        end_effectors: tuple[EndEffectorTarget, ...] = (),
    ) -> torch.Tensor:
        text_feat, text_pad_mask = prepare_conditioning(
            embedding,
            device=self.device,
        )
        generated_frames = self.fps * self.duration_s
        num_frames = generated_frames + self.history_frames
        lengths = torch.tensor([num_frames], device=self.device)
        motion_mask = observed_motion = None
        if target_xys or end_effectors:
            motion_mask, observed_motion = build_constraints(
                self.model.motion_rep,
                self.root_history,
                self.root_heading,
                target_xys,
                end_effectors,
                generated_frames=generated_frames,
                history_frames=self.history_frames,
                device=self.device,
            )

        with torch.inference_mode():
            motion = self.model(
                num_frames,
                num_denoising_steps=int(self.model.diffusion.num_base_steps),
                pad_mask=length_to_mask(lengths),
                first_heading_angle=None,
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                cfg_weight=(2.0, 2.0),
                progress_bar=lambda iterable: iterable,
                init_history_sequence=self.initial_history,
            )
            generated_motion = motion[:, self.history_frames :]
            if generated_motion.shape[1] != generated_frames:
                raise ValueError(
                    f"ARDY generated {generated_motion.shape[1]} frames; "
                    f"expected {generated_frames}"
                )
            next_history = motion[:, -self.history_frames :].detach().clone()
            decoded = self.model.motion_rep.inverse(
                generated_motion,
                is_normalized=True,
            )
            if end_effectors:
                decoded_pos = self.model.motion_rep.inverse(
                    generated_motion,
                    is_normalized=True,
                    posed_joints_from="positions",
                )
                self._log_end_effector_diagnostic(
                    generated_motion,
                    decoded,
                    decoded_pos,
                    end_effectors,
                )
            root_positions = decoded["root_positions"]
            root_headings = decoded["global_root_heading"]
            if root_positions.shape[1] < 2:
                raise ValueError("ARDY generated fewer than two root positions")
            if root_headings.ndim != 3 or root_headings.shape[-1] != 2:
                raise ValueError("ARDY global_root_heading must have shape [B, T, 2]")
            next_root_history = root_positions[0, -2:].detach().clone()
            final_heading = root_headings[0, -1]
            next_root_heading = (
                torch.atan2(final_heading[1], final_heading[0]).detach().clone()
            )
            batched_qpos = self.converter.dict_to_qpos(
                decoded,
                str(self.device),
                numpy=False,
            )

        qpos = batched_qpos[0].contiguous()
        self.initial_history = next_history
        self.root_history = next_root_history
        self.root_heading = next_root_heading
        return qpos

    def _log_end_effector_diagnostic(
        self,
        generated_motion: torch.Tensor,
        decoded_rot: dict[str, torch.Tensor],
        decoded_pos: dict[str, torch.Tensor],
        end_effectors: tuple[EndEffectorTarget, ...],
    ) -> None:
        """Temporary diagnostic comparing ARDY position and rotation features."""
        current_root = self.root_history[-1].to(device=self.device)
        heading = self.root_heading.to(device=self.device)
        target_positions = end_effector_target_positions(
            current_root,
            heading,
            end_effectors,
            device=self.device,
        )
        final_frame = generated_motion.shape[1] - 1
        position_joints = decoded_pos["posed_joints"][0]
        rotation_joints = decoded_rot["posed_joints"][0]
        joint_indices = [
            self.model.motion_rep.skeleton.bone_order_names.index(_JOINT_NAMES[target.name])
            for target in end_effectors
        ]

        for target, joint_index, target_position in zip(
            end_effectors, joint_indices, target_positions, strict=True
        ):
            position_feature = position_joints[final_frame, joint_index]
            rotation_fk = rotation_joints[final_frame, joint_index]
            target_to_position = torch.linalg.vector_norm(
                target_position - position_feature
            )
            target_to_fk = torch.linalg.vector_norm(target_position - rotation_fk)
            position_to_fk = torch.linalg.vector_norm(position_feature - rotation_fk)
            _LOGGER.warning(
                "ARDY EE diagnostic: %s\n"
                "  target:        %s\n"
                "  position_rep:  %s\n"
                "  rotation_fk:   %s\n"
                "  target->pos:   %.3f m\n"
                "  target->fk:    %.3f m\n"
                "  pos->fk:       %.3f m",
                target.name,
                _format_ardy_vector(target_position),
                _format_ardy_vector(position_feature),
                _format_ardy_vector(rotation_fk),
                float(target_to_position),
                float(target_to_fk),
                float(position_to_fk),
            )

            window_start = max(0, final_frame - 24)
            window_rotation = rotation_joints[window_start : final_frame + 1, joint_index]
            window_errors = torch.linalg.vector_norm(
                target_position.unsqueeze(0) - window_rotation, dim=1
            )
            _LOGGER.warning(
                "ARDY EE diagnostic window: %s final_25_frames=%d "
                "target->fk_by_frame=%s target->fk_min=%.3f m "
                "target->fk_final=%.3f m",
                target.name,
                window_errors.shape[0],
                _format_ardy_errors(window_errors),
                float(window_errors.min()),
                float(window_errors[-1]),
            )


def _format_ardy_vector(value: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(component):.4f}" for component in value) + "]"


def _format_ardy_errors(value: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(error):.4f}" for error in value) + "]"
