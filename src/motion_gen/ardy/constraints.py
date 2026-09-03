from __future__ import annotations

import torch
from ardy.motion_rep.tools import RotateFeatures

from shared.messages import EndEffectorTarget

_JOINT_NAMES = {
    "left_hand": "left_hand_roll_skel",
    "right_hand": "right_hand_roll_skel",
    "left_foot": "left_toe_base",
    "right_foot": "right_toe_base",
}
_MAX_END_EFFECTOR_REACH_M = {
    "left_hand": 1.25,
    "right_hand": 1.25,
    "left_foot": 1.5,
    "right_foot": 1.5,
}


def build_constraints(
    motion_rep,
    root_history: torch.Tensor,
    root_heading: torch.Tensor,
    target_xys: tuple[tuple[float, float], ...],
    end_effectors: tuple[EndEffectorTarget, ...],
    *,
    generated_frames: int,
    history_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert robot-local navigation and end-effector targets into ARDY conditions."""
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

    index: dict[str, list[torch.Tensor]] = {}
    data: dict[str, list[torch.Tensor]] = {}

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
    else:
        # ARDY's global-joint conditioning requires a Hips/root observation.
        # Pin an EE-only request to the current pelvis pose at its terminal frame.
        root_2d = current_root_2d.unsqueeze(0)
        frame_indices = torch.tensor(
            [generated_frames + history_frames - 1], device=device
        )
    index["root_2d"] = [frame_indices]
    data["root_2d"] = [root_2d]
    if end_effectors:
        frame = generated_frames + history_frames - 1
        target_positions = global_end_effector_targets(
            current_root, heading, end_effectors, device=device
        )
        root_index = motion_rep.skeleton.root_idx
        constraint_root = current_root.clone()
        constraint_root[[0, 2]] = root_2d[-1]
        _validate_end_effector_reach(end_effectors, target_positions, constraint_root)
        joint_indices = [
            motion_rep.skeleton.bone_order_names.index(_JOINT_NAMES[target.name])
            for target in end_effectors
        ]

        global_indices = torch.tensor(
            [[frame, root_index], *[[frame, joint] for joint in joint_indices]],
            device=device,
        )
        index.update(
            root_y_pos=[torch.tensor([frame], device=device)],
            global_root_heading=[torch.tensor([frame], device=device)],
            global_joints_positions=[global_indices],
        )
        data.update(
            root_y_pos=[current_root[1].reshape(1)],
            global_root_heading=[
                torch.stack((torch.cos(heading), torch.sin(heading))).unsqueeze(0)
            ],
            global_joints_positions=[
                torch.cat((constraint_root.unsqueeze(0), target_positions))
            ],
        )

    observed_motion, motion_mask = motion_rep.create_conditions(
        index,
        data,
        generated_frames + history_frames,
        True,
        device,
    )
    return motion_mask.unsqueeze(0), observed_motion.unsqueeze(0)


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


def _validate_end_effector_reach(
    end_effectors: tuple[EndEffectorTarget, ...],
    target_positions: torch.Tensor,
    final_root: torch.Tensor,
) -> None:
    """Validate reach from ARDY's final root constraint, not its current pose."""
    distances = torch.linalg.vector_norm(target_positions - final_root, dim=1)
    for target, distance in zip(end_effectors, distances, strict=True):
        limit = _MAX_END_EFFECTOR_REACH_M[target.name]
        distance_value = float(distance)
        if distance_value > limit:
            raise ValueError(
                f"{target.name} target is out of reach at the final waypoint: "
                f"distance={distance_value:.3f}m, maximum={limit:.3f}m"
            )
