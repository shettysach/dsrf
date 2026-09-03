from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import load_model

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
        text_cfg_weight: float = 2.0,
        constraint_cfg_weight: float = 2.0,
        seed: int | None = None,
    ) -> None:
        if text_cfg_weight < 0.0:
            raise ValueError("text_cfg_weight must be non-negative")
        if constraint_cfg_weight < 0.0:
            raise ValueError("constraint_cfg_weight must be non-negative")
        self.device = torch.device(device)
        self.text_cfg_weight = text_cfg_weight
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
        frames_per_token = int(self.model.num_frames_per_token)
        generated_frames = int(self.model.gen_horizon_len)
        if generated_frames % frames_per_token != 0:
            raise ValueError(
                "ARDY generation horizon must be a multiple of its token patch size"
            )
        # Match ARDY's interactive demo: retain exactly one tokenizer patch.
        self.history_crop_frames = frames_per_token
        # ARDY's interactive path starts the first window without a motion
        # history.  A history is only retained after that first generation and
        # used to continue later windows.
        self.motion_history: torch.Tensor | None = None
        standing_pose = np.repeat(standing_qpos()[None], 2, axis=0)
        _, root_positions = qpos_to_ardy_inputs(
            standing_pose,
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
        has_spatial_constraints = bool(target_xys or end_effectors)
        history_frames = (
            0 if self.motion_history is None else self.motion_history.shape[1]
        )
        generated_frames = int(self.model.gen_horizon_len)
        num_frames = history_frames + generated_frames
        root_motion_mask = root_observed_motion = None
        if target_xys:
            root_motion_mask, root_observed_motion = build_constraints(
                self.model.motion_rep,
                self.root_history,
                self.root_heading,
                target_xys,
                (),
                generated_frames=generated_frames,
                history_frames=history_frames,
                device=self.device,
            )

        with torch.inference_mode():
            cfg_weight = (
                (
                    getattr(self, "text_cfg_weight", 2.0),
                    getattr(self, "constraint_cfg_weight", 2.0),
                )
                if has_spatial_constraints
                else getattr(self, "text_cfg_weight", 2.0)
            )
            autoregressive_kwargs: dict[str, object] = {}
            if self.motion_history is None:
                # ARDY's initial translation is a horizontal-world offset;
                # root height belongs to the generated root motion itself.
                init_global_translation = self.root_history[-1:].clone()
                init_global_translation[:, 1] = 0.0
                autoregressive_kwargs.update(
                    init_global_translation=init_global_translation,
                    init_first_heading_angle=self.root_heading.reshape(1).to(self.device),
                )
            else:
                autoregressive_kwargs["init_history_sequence"] = self.motion_history

            def generate_window(
                motion_mask: torch.Tensor | None,
                observed_motion: torch.Tensor | None,
            ) -> torch.Tensor:
                # Reset immediately before every pass so reference and final
                # generation use identical diffusion noise.
                seed = getattr(self, "seed", None)
                if seed is not None:
                    torch.manual_seed(seed)
                return self.model.autoregressive_step(
                    num_frames=num_frames,
                    num_denoising_steps=int(self.model.diffusion.num_base_steps),
                    motion_mask=motion_mask,
                    observed_motion=observed_motion,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                    cfg_weight=cfg_weight,
                    **autoregressive_kwargs,
                )

            if end_effectors:
                reference_motion = generate_window(
                    root_motion_mask, root_observed_motion
                )
                reference_generated = reference_motion[:, history_frames:]
                reference_decoded = self.model.motion_rep.inverse(
                    reference_generated,
                    is_normalized=True,
                )
                motion_mask, observed_motion = build_constraints(
                    self.model.motion_rep,
                    self.root_history,
                    self.root_heading,
                    target_xys,
                    end_effectors,
                    reference_decoded,
                    generated_frames=generated_frames,
                    history_frames=history_frames,
                    device=self.device,
                )
                motion = generate_window(motion_mask, observed_motion)
            else:
                motion = generate_window(root_motion_mask, root_observed_motion)
            if motion.shape[1] != num_frames:
                raise ValueError(
                    f"ARDY returned {motion.shape[1]} total frames; expected {num_frames}"
                )
            generated_motion = motion[:, history_frames:]
            if generated_motion.shape[1] != generated_frames:
                raise ValueError(
                    f"ARDY generated {generated_motion.shape[1]} frames; "
                    f"expected {generated_frames}"
                )
            next_history = motion[:, -self.history_crop_frames :].detach().clone()
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
        self.motion_history = next_history
        self.root_history = next_root_history
        self.root_heading = next_root_heading
        return qpos
