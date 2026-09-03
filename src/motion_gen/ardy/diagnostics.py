"""Controlled ARDY constraint-conditioning diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from motion_gen.ardy.constraints import build_constraints, global_end_effector_targets
from motion_gen.ardy.encoder import prepare_conditioning
from shared.messages import EndEffectorTarget

if TYPE_CHECKING:
    from motion_gen.ardy.generator import Ardy


@dataclass(frozen=True)
class EEConditionComparison:
    """Raw outputs from matched unconstrained and EE-conditioned ARDY calls."""

    unconstrained_motion: torch.Tensor
    conditioned_motion: torch.Tensor
    motion_mask: torch.Tensor
    observed_motion: torch.Tensor
    unconstrained_decoded: dict[str, torch.Tensor]
    conditioned_decoded: dict[str, torch.Tensor]
    target_positions: torch.Tensor


def compare_first_generation_ee_conditioning(
    generator: Ardy,
    embedding: torch.Tensor,
    end_effectors: tuple[EndEffectorTarget, ...],
) -> EEConditionComparison:
    """Run a seed-matched first-window A/B comparison.

    Both calls have no ARDY motion history and use the same text/constraint CFG
    tuple, root translation, heading, denoising steps, and random seed.  Only
    ``motion_mask`` and ``observed_motion`` differ.
    """
    if generator.motion_history is not None:
        raise ValueError("EE comparison requires a fresh generator with no history")
    if not end_effectors:
        raise ValueError("EE comparison requires at least one end-effector target")

    text_feat, text_pad_mask = prepare_conditioning(embedding, device=generator.device)
    generated_frames = int(generator.model.gen_horizon_len)
    motion_mask, observed_motion = build_constraints(
        generator.model.motion_rep,
        generator.root_history,
        generator.root_heading,
        (),
        end_effectors,
        generated_frames=generated_frames,
        history_frames=0,
        device=generator.device,
    )
    init_global_translation = generator.root_history[-1:].clone()
    init_global_translation[:, 1] = 0.0
    init_first_heading_angle = generator.root_heading.reshape(1).to(generator.device)
    cfg_weight = (
        getattr(generator, "text_cfg_weight", 2.0),
        getattr(generator, "constraint_cfg_weight", 2.0),
    )

    def generate(mask: torch.Tensor | None, observed: torch.Tensor | None) -> torch.Tensor:
        seed = getattr(generator, "seed", None)
        if seed is not None:
            torch.manual_seed(seed)
        return generator.model.autoregressive_step(
            num_frames=generated_frames,
            num_denoising_steps=int(generator.model.diffusion.num_base_steps),
            motion_mask=mask,
            observed_motion=observed,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            cfg_weight=cfg_weight,
            init_global_translation=init_global_translation,
            init_first_heading_angle=init_first_heading_angle,
        )

    with torch.inference_mode():
        unconstrained_motion = generate(None, None)
        conditioned_motion = generate(motion_mask, observed_motion)
        if unconstrained_motion.shape[1] != generated_frames:
            raise ValueError("Unconstrained ARDY call returned an unexpected frame count")
        if conditioned_motion.shape != unconstrained_motion.shape:
            raise ValueError("Conditioned and unconstrained ARDY outputs differ in shape")
        unconstrained_decoded = generator.model.motion_rep.inverse(
            unconstrained_motion, is_normalized=True
        )
        conditioned_decoded = generator.model.motion_rep.inverse(
            conditioned_motion, is_normalized=True
        )
        target_positions = global_end_effector_targets(
            generator.root_history[-1],
            generator.root_heading.to(generator.device),
            end_effectors,
            device=generator.device,
        )

    return EEConditionComparison(
        unconstrained_motion=unconstrained_motion,
        conditioned_motion=conditioned_motion,
        motion_mask=motion_mask,
        observed_motion=observed_motion,
        unconstrained_decoded=unconstrained_decoded,
        conditioned_decoded=conditioned_decoded,
        target_positions=target_positions,
    )
