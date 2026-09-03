"""Controlled ARDY constraint-conditioning diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from motion_gen.ardy.constraints import (
    build_constraints,
    global_end_effector_targets,
    native_constraint_source,
)
from motion_gen.ardy.encoder import prepare_conditioning
from shared.messages import EndEffectorTarget

if TYPE_CHECKING:
    from motion_gen.ardy.generator import Ardy


@dataclass(frozen=True)
class EEConditionComparison:
    """Raw outputs from matched unconstrained, sparse, and native EE calls."""

    unconstrained_motion: torch.Tensor
    conditioned_motion: torch.Tensor
    motion_mask: torch.Tensor
    observed_motion: torch.Tensor
    unconstrained_decoded: dict[str, torch.Tensor]
    conditioned_decoded: dict[str, torch.Tensor]
    native_motion: torch.Tensor
    native_motion_mask: torch.Tensor
    native_observed_motion: torch.Tensor
    native_decoded: dict[str, torch.Tensor]
    native_constraint_source: str
    target_positions: torch.Tensor


def compare_first_generation_ee_conditioning(
    generator: Ardy,
    embedding: torch.Tensor,
    end_effectors: tuple[EndEffectorTarget, ...],
) -> EEConditionComparison:
    """Run seed-matched A/B/C generation against the requested hand XYZ."""
    if generator.motion_history is not None:
        raise ValueError("EE comparison requires a fresh generator with no history")
    if len(end_effectors) != 1 or end_effectors[0].name not in {
        "left_hand",
        "right_hand",
    }:
        raise ValueError("EE comparison supports exactly one hand target")

    text_feat, text_pad_mask = prepare_conditioning(embedding, device=generator.device)
    generated_frames = int(generator.model.gen_horizon_len)
    sparse_mask, sparse_observed = _legacy_sparse_ee_constraints(
        generator.model.motion_rep,
        generator.root_history[-1],
        generator.root_heading,
        end_effectors,
        generated_frames=generated_frames,
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
        unconstrained_decoded = generator.model.motion_rep.inverse(
            unconstrained_motion, is_normalized=True
        )
        sparse_motion = generate(sparse_mask, sparse_observed)
        sparse_decoded = generator.model.motion_rep.inverse(
            sparse_motion, is_normalized=True
        )
        native_mask, native_observed = build_constraints(
            generator.model.motion_rep,
            generator.root_history,
            generator.root_heading,
            (),
            end_effectors,
            unconstrained_decoded,
            generated_frames=generated_frames,
            history_frames=0,
            device=generator.device,
        )
        native_motion = generate(native_mask, native_observed)
        native_decoded = generator.model.motion_rep.inverse(
            native_motion, is_normalized=True
        )
        target_positions = global_end_effector_targets(
            generator.root_history[-1],
            generator.root_heading.to(generator.device),
            end_effectors,
            device=generator.device,
        )

    if not (
        unconstrained_motion.shape == sparse_motion.shape == native_motion.shape
    ):
        raise ValueError("ARDY diagnostic outputs differ in shape")
    return EEConditionComparison(
        unconstrained_motion=unconstrained_motion,
        conditioned_motion=sparse_motion,
        motion_mask=sparse_mask,
        observed_motion=sparse_observed,
        unconstrained_decoded=unconstrained_decoded,
        conditioned_decoded=sparse_decoded,
        native_motion=native_motion,
        native_motion_mask=native_mask,
        native_observed_motion=native_observed,
        native_decoded=native_decoded,
        native_constraint_source=native_constraint_source(),
        target_positions=target_positions,
    )


def _legacy_sparse_ee_constraints(
    motion_rep,
    current_root: torch.Tensor,
    root_heading: torch.Tensor,
    end_effectors: tuple[EndEffectorTarget, ...],
    *,
    generated_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the retired 3-feature mask for diagnostic arm B only."""
    targets = global_end_effector_targets(
        current_root,
        root_heading.to(device),
        end_effectors,
        device=device,
    )
    observed = torch.zeros(
        (generated_frames, motion_rep.motion_rep_dim),
        dtype=current_root.dtype,
        device=device,
    )
    mask = torch.zeros_like(observed, dtype=torch.bool)
    joint_slice = motion_rep.slice_dict["local_joints_positions"]
    positions = observed[:, joint_slice].reshape(
        generated_frames, motion_rep.skeleton.nbjoints - 1, 3
    )
    position_mask = mask[:, joint_slice].reshape_as(positions)
    for target, target_position in zip(end_effectors, targets, strict=True):
        side = target.name.removesuffix("_hand")
        joint = motion_rep.skeleton.bone_order_names.index(
            f"{side}_hand_roll_skel"
        )
        local_position = target_position - current_root.to(device)
        local_position[1] = target_position[1]
        positions[-1, joint - 1] = local_position
        position_mask[-1, joint - 1] = True
    normalized = motion_rep.normalize(observed) * mask
    return mask.unsqueeze(0).float(), normalized.unsqueeze(0)
