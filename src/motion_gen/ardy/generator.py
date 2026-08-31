from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import load_model
from ardy.motion_rep.tools import length_to_mask

from motion_gen.ardy.constraints import build_constraints
from motion_gen.ardy.encoder import prepare_conditioning
from motion_gen.ardy.history import qpos_to_ardy_inputs
from shared.g1 import standing_qpos
from shared.messages import EndEffectorTarget


class Ardy:
    """Text-conditioned ARDY motion generator for Unitree G1."""

    fps = 25

    def __init__(
        self,
        checkpoints_dir: Path,
        *,
        device: str = "cpu",
        constraint_cfg_weight: float = 2.0,
        seed: int | None = None,
    ) -> None:
        if constraint_cfg_weight < 0.0:
            raise ValueError("constraint_cfg_weight must be non-negative")
        self.device = torch.device(device)
        self.constraint_cfg_weight = constraint_cfg_weight
        self.seed = seed
        self.model = load_model(
            "g1",
            device=str(self.device),
            checkpoints_dir=str(checkpoints_dir),
        )
        model_fps = float(self.model.motion_rep.fps)
        if model_fps != self.fps:
            raise ValueError(f"Expected ARDY G1 at {self.fps} FPS, got {model_fps}")

        self.converter = MujocoQposConverter(self.model.skeleton)
        self.root_history: torch.Tensor | None = None
        self.root_heading: torch.Tensor | None = None

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
        has_spatial_constraints = bool(target_xys or end_effectors)
        generated_frames = int(self.model.gen_horizon_len)
        lengths = torch.tensor([generated_frames], device=self.device)
        motion_mask = observed_motion = None
        if has_spatial_constraints:
            root_history = (
                self.root_history
                if self.root_history is not None
                else self._standing_constraint_root_history()
            )
            root_heading = (
                self.root_heading
                if self.root_heading is not None
                else torch.tensor(0.0, device=self.device)
            )
            motion_mask, observed_motion = build_constraints(
                self.model.motion_rep,
                root_history,
                root_heading,
                target_xys,
                end_effectors,
                generated_frames=generated_frames,
                history_frames=0,
                device=self.device,
            )

        with torch.inference_mode():
            # Seed immediately before diffusion so separate process runs use
            # identical noise, regardless of model-loading side effects.
            seed = getattr(self, "seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            motion = self.model(
                generated_frames,
                num_denoising_steps=int(self.model.diffusion.num_base_steps),
                pad_mask=length_to_mask(lengths),
                first_heading_angle=torch.zeros(1, device=self.device),
                motion_mask=motion_mask,
                observed_motion=observed_motion,
                text_feat=text_feat,
                text_pad_mask=text_pad_mask,
                cfg_weight=(2.0, getattr(self, "constraint_cfg_weight", 2.0))
                if has_spatial_constraints
                else 2.0,
                progress_bar=lambda iterable: iterable,
                init_history_sequence=None,
            )
            generated_motion = motion
            if generated_motion.shape[1] != generated_frames:
                raise ValueError(
                    f"ARDY generated {generated_motion.shape[1]} frames; "
                    f"expected {generated_frames}"
                )
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
        self.root_history = next_root_history
        self.root_heading = next_root_heading
        return qpos

    def _standing_constraint_root_history(self) -> torch.Tensor:
        """Return the standing pose only as a spatial coordinate reference."""
        standing_qpos_history = np.repeat(standing_qpos()[None], 2, axis=0)
        _, root_positions = qpos_to_ardy_inputs(
            standing_qpos_history,
            self.converter,
            device=self.device,
        )
        return root_positions[0, -2:].detach().clone()
