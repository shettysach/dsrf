from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import load_model
from ardy.motion_rep.tools import length_to_mask

from motion_gen.ardy.constraints import build_constraints
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
    ) -> np.ndarray:
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
            root_positions = decoded["root_positions"]
            root_headings = decoded["global_root_heading"]
            if root_positions.shape[1] < 2:
                raise ValueError("ARDY generated fewer than two root positions")
            if root_headings.ndim != 3 or root_headings.shape[-1] != 2:
                raise ValueError("ARDY global_root_heading must have shape [B, T, 2]")
            next_root_history = root_positions[0, -2:].detach().clone()
            final_heading = root_headings[0, -1]
            if (
                not torch.isfinite(final_heading).all()
                or torch.linalg.vector_norm(final_heading) <= 1e-8
            ):
                raise ValueError("ARDY generated an invalid root heading")
            next_root_heading = (
                torch.atan2(final_heading[1], final_heading[0]).detach().clone()
            )
            batched_qpos = self.converter.dict_to_qpos(
                decoded,
                str(self.device),
            )

        qpos = np.ascontiguousarray(batched_qpos[0], dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"ARDY qpos must have shape [T, 36], got {qpos.shape}")
        if qpos.shape[0] == 0:
            raise ValueError("ARDY generated no motion frames")
        if not np.isfinite(qpos).all():
            raise ValueError("ARDY qpos contains NaN or infinite values")
        self.initial_history = next_history
        self.root_history = next_root_history
        self.root_heading = next_root_heading
        return qpos
