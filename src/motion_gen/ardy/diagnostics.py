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
    """Run a seed-matched first-window A/B comparison.

    Both calls have no ARDY motion history and use the same text/constraint CFG
    tuple, root translation, heading, denoising steps, and random seed.  Only
    ``motion_mask`` and ``observed_motion`` differ.
    """
    if generator.motion_history is not None:
        raise ValueError("EE comparison requires a fresh generator with no history")
    if not end_effectors:
        raise ValueError("EE comparison requires at least one end-effector target")
    if len(end_effectors) != 1 or end_effectors[0].name not in {
        "left_hand",
        "right_hand",
    }:
        raise ValueError("Native EE comparison supports exactly one hand target")

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
        native_constraint, native_constraint_source = _native_hand_constraint(
            generator.model.skeleton,
            end_effectors[0].name,
            unconstrained_decoded["posed_joints"][:, -1],
            unconstrained_decoded["global_rot_mats"][:, -1],
            generated_frames - 1,
        )
        native_observed_motion, native_motion_mask = (
            generator.model.motion_rep.create_conditions_from_constraints(
                [native_constraint],
                generated_frames,
                False,
                str(generator.device),
            )
        )
        native_observed_motion = (
            generator.model.motion_rep.normalize(native_observed_motion)
            * native_motion_mask
        ).unsqueeze(0)
        native_motion_mask = native_motion_mask.unsqueeze(0)
        native_motion = generate(native_motion_mask, native_observed_motion)
        native_decoded = generator.model.motion_rep.inverse(
            native_motion, is_normalized=True
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
        native_motion=native_motion,
        native_motion_mask=native_motion_mask,
        native_observed_motion=native_observed_motion,
        native_decoded=native_decoded,
        native_constraint_source=native_constraint_source,
        target_positions=target_positions,
    )


def _native_hand_constraint(
    skeleton,
    hand: str,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    frame_index: int,
) -> tuple[object, str]:
    """Use ARDY's hand class when available, else its source-identical shim.

    The dependency pinned by this project predates ``ardy.constraints`` but
    retains ``create_conditions_from_constraints``.  The shim exists solely to
    make the A/B/C probe runnable against that pinned version.
    """
    try:
        from ardy.constraints import LeftHandConstraintSet, RightHandConstraintSet
    except ImportError:
        classes = {"left_hand": _NativeHandConstraint, "right_hand": _NativeHandConstraint}
        return (
            classes[hand](skeleton, positions, rotations, hand, frame_index),
            "source-identical compatibility shim",
        )
    constraint_class = (
        LeftHandConstraintSet if hand == "left_hand" else RightHandConstraintSet
    )
    frame = torch.tensor([frame_index], device=positions.device)
    return (
        constraint_class(skeleton, frame, positions, rotations, root_2d=None),
        "ardy.constraints",
    )


class _NativeHandConstraint:
    """Minimal implementation of NVIDIA's EndEffectorConstraintSet update path."""

    def __init__(
        self,
        skeleton,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        hand: str,
        frame_index: int,
    ):
        from ardy.motion_rep.tools import compute_heading_angle

        self.skeleton = skeleton
        self.positions = positions
        self.rotations = rotations
        self.frame = torch.tensor([frame_index], device=positions.device)
        hand_name = "LeftHand" if hand == "left_hand" else "RightHand"
        rot_names, pos_names = skeleton.expand_joint_names([hand_name, "Hips"])
        self.rot_indices = torch.tensor(
            [skeleton.bone_index[name] for name in rot_names], device=positions.device
        )
        self.pos_indices = torch.tensor(
            [skeleton.bone_index[name] for name in pos_names], device=positions.device
        )
        heading = compute_heading_angle(positions, skeleton)
        self.global_root_heading = torch.stack(
            (torch.cos(heading), torch.sin(heading)), dim=-1
        )

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        index_dict["global_joints_positions"].append(
            torch.stack(
                (self.frame.expand(len(self.pos_indices)), self.pos_indices), dim=-1
            )
        )
        data_dict["global_joints_positions"].append(self.positions[0, self.pos_indices])
        index_dict["global_joints_rots"].append(
            torch.stack(
                (self.frame.expand(len(self.rot_indices)), self.rot_indices), dim=-1
            )
        )
        data_dict["global_joints_rots"].append(self.rotations[0, self.rot_indices])
        root = self.positions[:, self.skeleton.root_idx]
        index_dict["root_2d"].append(self.frame)
        data_dict["root_2d"].append(root[:, [0, 2]])
        index_dict["root_y_pos"].append(self.frame)
        data_dict["root_y_pos"].append(root[:, 1])
        index_dict["global_root_heading"].append(self.frame)
        data_dict["global_root_heading"].append(self.global_root_heading)
