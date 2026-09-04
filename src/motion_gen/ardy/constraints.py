import torch
from ardy.constraints import (
    LeftFootConstraintSet,
    LeftHandConstraintSet,
    RightFootConstraintSet,
    RightHandConstraintSet,
    Root2DConstraintSet,
)
from ardy.motion_rep.tools import RotateFeatures

from shared.messages import EndEffectorTarget

_JOINT_NAMES = {
    "left_hand": "left_hand_roll_skel",
    "right_hand": "right_hand_roll_skel",
    "left_foot": "left_toe_base",
    "right_foot": "right_toe_base",
}


def build_constraints(
    motion_rep,
    root_history: torch.Tensor,
    root_heading: torch.Tensor,
    target_xys: tuple[tuple[float, float], ...],
    end_effectors: tuple[EndEffectorTarget, ...],
    reference_decoded: dict[str, torch.Tensor] | None = None,
    *,
    generated_frames: int,
    history_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build root and end-effector constraints for an ARDY generation."""
    if (
        root_history.ndim != 2
        or root_history.shape[0] < 2
        or root_history.shape[1] != 3
    ):
        raise ValueError(
            "ARDY root history must have shape [T >= 2, 3], "
            f"got {tuple(root_history.shape)}"
        )

    current_root = root_history[-1].to(device=device)
    current_root_2d = current_root[[0, 2]]
    if root_heading.numel() != 1:
        raise ValueError("ARDY root heading must contain one angle")
    if not target_xys and not end_effectors:
        raise ValueError("ARDY constraints require at least one target")
    heading = root_heading.reshape(()).to(dtype=current_root_2d.dtype, device=device)

    constraints: list[object] = []
    if target_xys:
        local_2d = torch.tensor(
            [[left, forward] for forward, left in target_xys],
            dtype=current_root.dtype,
            device=device,
        )
        root_2d = current_root_2d + _rotate_2d(local_2d, heading)
        relative_indices = (
            torch.arange(1, len(target_xys) + 1, device=device)
            * generated_frames
            // len(target_xys)
        )
        frame_indices = relative_indices + history_frames - 1
        constraints.append(_root_constraint(motion_rep.skeleton, frame_indices, root_2d))

    if end_effectors:
        if reference_decoded is None:
            raise ValueError("ARDY end-effector constraints require a reference pose")
        target_positions = global_end_effector_targets(
            current_root, heading, end_effectors, device=device
        )
        positions, rotations = _reference_final_pose(reference_decoded, device)
        frame = torch.tensor(
            [generated_frames + history_frames - 1], device=device
        )
        for target, target_position in zip(
            end_effectors, target_positions, strict=True
        ):
            edited_positions = positions.clone()
            hand_index = motion_rep.skeleton.bone_order_names.index(
                _JOINT_NAMES[target.name]
            )
            delta = target_position - edited_positions[0, hand_index]
            base_name = _base_end_effector_name(target.name)
            _, chain_names = motion_rep.skeleton.expand_joint_names([base_name])
            chain_indices = [
                motion_rep.skeleton.bone_order_names.index(name)
                for name in chain_names
            ]
            edited_positions[0, chain_indices] += delta
            constraints.append(
                _end_effector_constraint(
                    motion_rep.skeleton,
                    target.name,
                    frame,
                    edited_positions,
                    rotations,
                )
            )

    observed_motion, motion_mask = motion_rep.create_conditions_from_constraints(
        constraints,
        generated_frames + history_frames,
        False,
        str(device),
    )
    normalized = motion_rep.normalize(observed_motion) * motion_mask
    return motion_mask.unsqueeze(0), normalized.unsqueeze(0)


def _reference_final_pose(
    decoded: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        positions = decoded["posed_joints"][:, -1].to(device=device)
        rotations = decoded["global_rot_mats"][:, -1].to(device=device)
    except KeyError as exc:
        raise ValueError(f"ARDY reference pose is missing {exc.args[0]}") from exc
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("ARDY posed_joints must have shape [B, T, J, 3]")
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("ARDY global_rot_mats must have shape [B, T, J, 3, 3]")
    if positions.shape[:2] != rotations.shape[:2] or positions.shape[0] != 1:
        raise ValueError("ARDY reference positions and rotations must share one pose")
    return positions, rotations


def _base_end_effector_name(name: str) -> str:
    return {
        "left_hand": "LeftHand",
        "right_hand": "RightHand",
        "left_foot": "LeftFoot",
        "right_foot": "RightFoot",
    }[name]


def _root_constraint(skeleton, frame_indices: torch.Tensor, root_2d: torch.Tensor):
    return Root2DConstraintSet(skeleton, frame_indices, root_2d)


def _end_effector_constraint(
    skeleton,
    name: str,
    frame_indices: torch.Tensor,
    positions: torch.Tensor,
    rotations: torch.Tensor,
):
    constraint_class = {
        "left_hand": LeftHandConstraintSet,
        "right_hand": RightHandConstraintSet,
        "left_foot": LeftFootConstraintSet,
        "right_foot": RightFootConstraintSet,
    }[name]
    return constraint_class(
        skeleton,
        frame_indices=frame_indices,
        global_joints_positions=positions,
        global_joints_rots=rotations,
        root_2d=None,
    )


def global_end_effector_targets(
    root_position: torch.Tensor,
    root_heading: torch.Tensor,
    end_effectors: tuple[EndEffectorTarget, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert robot-local end-effector targets to ARDY global XYZ coordinates."""
    local_2d = torch.tensor(
        [[target.target_xyz[1], target.target_xyz[0]] for target in end_effectors],
        dtype=root_position.dtype,
        device=device,
    )
    delta_2d = _rotate_2d(local_2d, root_heading.reshape(()))
    target_positions = root_position.to(device=device).repeat(len(end_effectors), 1)
    target_positions[:, [0, 2]] += delta_2d
    target_positions[:, 1] += torch.tensor(
        [target.target_xyz[2] for target in end_effectors],
        dtype=root_position.dtype,
        device=device,
    )
    return target_positions


def _rotate_2d(positions: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    return (
        RotateFeatures(heading.unsqueeze(0))
        .rotate_2d_positions(positions.unsqueeze(0))
        .squeeze(0)
    )
