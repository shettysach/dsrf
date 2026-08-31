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
_END_EFFECTOR_APPROACH_FRACTION = 0.30


def build_constraints(
    motion_rep,
    root_history: torch.Tensor,
    root_heading: torch.Tensor,
    target_xys: tuple[tuple[float, float], ...],
    end_effectors: tuple[EndEffectorTarget, ...],
    end_effector_start_positions: torch.Tensor | None = None,
    end_effector_root_positions: torch.Tensor | None = None,
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
        index["root_2d"] = [frame_indices]
        data["root_2d"] = [root_2d]
    else:
        frame = generated_frames + history_frames - 1
        frame_indices = torch.tensor([frame], device=device)
        root_2d = current_root_2d.unsqueeze(0)
        index["root_2d"] = [frame_indices]
        data["root_2d"] = [root_2d]
    if end_effectors:
        final_frame = generated_frames + history_frames - 1
        local_2d = torch.tensor(
            [[target.target_xyz[1], target.target_xyz[0]] for target in end_effectors],
            dtype=current_root.dtype,
            device=device,
        )
        delta_2d = _rotate_2d(local_2d, heading)
        target_positions = current_root.repeat(len(end_effectors), 1)
        target_positions[:, [0, 2]] += delta_2d
        target_positions[:, 1] += torch.tensor(
            [target.target_xyz[2] for target in end_effectors],
            dtype=current_root.dtype,
            device=device,
        )
        root_index = motion_rep.skeleton.root_idx
        constraint_root = current_root.clone()
        constraint_root[[0, 2]] = root_2d[-1]
        _validate_end_effector_reach(end_effectors, target_positions, constraint_root)
        joint_indices = [
            motion_rep.skeleton.bone_order_names.index(_JOINT_NAMES[target.name])
            for target in end_effectors
        ]

        frames = torch.tensor([final_frame], device=device)
        positions = target_positions
        root_positions = constraint_root.unsqueeze(0)
        if end_effector_start_positions is not None:
            start_positions = end_effector_start_positions.to(
                dtype=current_root.dtype, device=device
            )
            expected_shape = target_positions.shape
            if start_positions.shape != expected_shape:
                raise ValueError(
                    "ARDY end-effector start positions must have shape "
                    f"{tuple(expected_shape)}, got {tuple(start_positions.shape)}"
                )
            approach_frames = max(
                2, round(generated_frames * _END_EFFECTOR_APPROACH_FRACTION)
            )
            start_frame = final_frame - approach_frames + 1
            frames = torch.arange(start_frame, final_frame + 1, device=device)
            alpha = torch.linspace(0, 1, approach_frames, device=device)
            alpha = alpha.square() * (3 - 2 * alpha)
            positions = (
                (1 - alpha[:, None, None]) * start_positions[None]
                + alpha[:, None, None] * target_positions[None]
            ).reshape(-1, 3)
            if end_effector_root_positions is None:
                root_positions = current_root.repeat(approach_frames, 1)
            else:
                root_positions = end_effector_root_positions.to(
                    dtype=current_root.dtype, device=device
                )
                expected_shape = (approach_frames, 3)
                if root_positions.shape != expected_shape:
                    raise ValueError(
                        "ARDY end-effector root positions must have shape "
                        f"{expected_shape}, got {tuple(root_positions.shape)}"
                    )
            # The final root remains the explicit waypoint constraint; the
            # preceding roots retain the unconditioned motion's trajectory.
            root_positions = root_positions.clone()
            root_positions[-1, [0, 2]] = constraint_root[[0, 2]]
            index["root_2d"] = [frames]
            data["root_2d"] = [root_positions[:, [0, 2]]]
        global_indices = torch.tensor(
            [
                *[[frame, root_index] for frame in frames.tolist()],
                *[
                [frame, joint]
                for frame in frames.tolist()
                for joint in joint_indices
                ],
            ],
            device=device,
        )
        index.update(
            root_y_pos=[frames],
            global_joints_positions=[global_indices],
        )
        data.update(
            root_y_pos=[root_positions[:, 1]],
            global_joints_positions=[
                torch.cat((root_positions, positions))
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
