from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import load_model
from ardy.motion_rep.reps.ardy_motionrep import fk
from ardy.motion_rep.tools import length_to_mask

from motion_gen.ardy.constraints import _rotate_2d, build_constraints
from motion_gen.ardy.encoder import prepare_conditioning
from motion_gen.ardy.history import build_initial_history, qpos_to_ardy_inputs
from shared.g1 import standing_qpos
from shared.messages import EndEffectorTarget


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
        end_effector_start_positions = None
        end_effector_root_positions = None
        if end_effectors:
            # Sample the gross motion first.  Its palm positions anchor a short,
            # smooth final approach instead of pulling the arms from frame zero.
            preliminary_mask = preliminary_observed = None
            if target_xys:
                preliminary_mask, preliminary_observed = build_constraints(
                    self.model.motion_rep,
                    self.root_history,
                    self.root_heading,
                    target_xys,
                    (),
                    generated_frames=generated_frames,
                    history_frames=self.history_frames,
                    device=self.device,
                )
            with torch.inference_mode():
                preliminary_motion = self.model(
                    num_frames,
                    num_denoising_steps=int(self.model.diffusion.num_base_steps),
                    pad_mask=length_to_mask(lengths),
                    first_heading_angle=None,
                    motion_mask=preliminary_mask,
                    observed_motion=preliminary_observed,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    cfg_weight=(2.0, 2.0),
                    progress_bar=lambda iterable: iterable,
                    init_history_sequence=self.initial_history,
                )
                preliminary_decoded = self.model.motion_rep.inverse(
                    preliminary_motion[:, self.history_frames :],
                    is_normalized=True,
                )
            approach_start = generated_frames - max(2, round(generated_frames * 0.30))
            joint_indices = [
                self.model.skeleton.bone_order_names.index(
                    {
                        "left_hand": "left_hand_roll_skel",
                        "right_hand": "right_hand_roll_skel",
                        "left_foot": "left_toe_base",
                        "right_foot": "right_toe_base",
                    }[target.name]
                )
                for target in end_effectors
            ]
            end_effector_start_positions = preliminary_decoded["posed_joints"][
                0, approach_start, joint_indices
            ].detach()
            end_effector_root_positions = preliminary_decoded["root_positions"][
                0, approach_start:
            ].detach()

        if target_xys or end_effectors:
            motion_mask, observed_motion = build_constraints(
                self.model.motion_rep,
                self.root_history,
                self.root_heading,
                target_xys,
                end_effectors,
                end_effector_start_positions,
                end_effector_root_positions,
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
        if end_effectors and os.getenv("ARDY_END_EFFECTOR_DIAGNOSTICS") == "1":
            self._log_end_effector_diagnostics(
                generated_motion,
                decoded,
                qpos,
                end_effectors,
                end_effector_start_positions,
                approach_start,
            )
            # MJLab captures its CUDA graph immediately after generation. Free
            # diagnostic-only temporaries from PyTorch's caching allocator first.
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.initial_history = next_history
        self.root_history = next_root_history
        self.root_heading = next_root_heading
        return qpos

    def _log_end_effector_diagnostics(
        self,
        generated_motion: torch.Tensor,
        decoded_from_rotations: dict[str, torch.Tensor],
        qpos: torch.Tensor,
        end_effectors: tuple[EndEffectorTarget, ...],
        start_positions: torch.Tensor,
        approach_start: int,
    ) -> None:
        """Print requested, feature-space, FK, and qpos-round-trip hand paths."""
        with torch.inference_mode():
            report = self._end_effector_diagnostic_report(
                generated_motion,
                decoded_from_rotations,
                qpos,
                end_effectors,
                start_positions,
                approach_start,
            )
        print("ARDY end-effector diagnostics=" + json.dumps(report))

    def _end_effector_diagnostic_report(
        self,
        generated_motion: torch.Tensor,
        decoded_from_rotations: dict[str, torch.Tensor],
        qpos: torch.Tensor,
        end_effectors: tuple[EndEffectorTarget, ...],
        start_positions: torch.Tensor,
        approach_start: int,
    ) -> list[dict[str, object]]:
        decoded_from_positions = self.model.motion_rep.inverse(
            generated_motion, is_normalized=True, posed_joints_from="positions"
        )
        names_to_joints = {
            "left_hand": "left_hand_roll_skel",
            "right_hand": "right_hand_roll_skel",
            "left_foot": "left_toe_base",
            "right_foot": "right_toe_base",
        }
        joint_indices = [
            self.model.skeleton.bone_order_names.index(names_to_joints[target.name])
            for target in end_effectors
        ]
        current_root = self.root_history[-1].to(device=self.device)
        local_2d = torch.tensor(
            [[target.target_xyz[1], target.target_xyz[0]] for target in end_effectors],
            dtype=current_root.dtype,
            device=self.device,
        )
        heading = self.root_heading.reshape(()).to(
            dtype=current_root.dtype, device=self.device
        )
        final_targets = current_root.repeat(len(end_effectors), 1)
        final_targets[:, [0, 2]] += _rotate_2d(local_2d, heading)
        final_targets[:, 1] += torch.tensor(
            [target.target_xyz[2] for target in end_effectors],
            dtype=current_root.dtype,
            device=self.device,
        )
        count = generated_motion.shape[1] - approach_start
        alpha = torch.linspace(0, 1, count, device=self.device)
        alpha = alpha.square() * (3 - 2 * alpha)
        requested = (1 - alpha[:, None, None]) * start_positions[None] + alpha[
            :, None, None
        ] * final_targets[None]
        local_rot_mats, qpos_root_positions = qpos_to_ardy_inputs(
            qpos.detach().cpu().numpy(), self.converter, device=self.device
        )
        _, qpos_roundtrip_positions, _ = fk(
            local_rot_mats[0], qpos_root_positions[0], self.model.skeleton
        )
        return [
            {
                "frame": approach_start + offset,
                "requested": requested[offset, target_index].tolist(),
                "position_features": decoded_from_positions["posed_joints"][
                    0, approach_start + offset, joint
                ].tolist(),
                "rotation_fk": decoded_from_rotations["posed_joints"][
                    0, approach_start + offset, joint
                ].tolist(),
                "qpos_roundtrip_fk": qpos_roundtrip_positions[
                    approach_start + offset, joint
                ].tolist(),
            }
            for offset in range(count)
            for target_index, joint in enumerate(joint_indices)
        ]
