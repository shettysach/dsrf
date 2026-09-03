from __future__ import annotations

import importlib

import torch
from ardy.motion_rep.tools import RotateFeatures, compute_heading_angle

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
    reference_decoded: dict[str, torch.Tensor] | None = None,
    *,
    generated_frames: int,
    history_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build root and native-style EE constraints for an ARDY generation."""
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
        final_root = positions[0, motion_rep.skeleton.root_idx]
        _validate_end_effector_reach(end_effectors, target_positions, final_root)
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
    try:
        constraints_module = importlib.import_module("ardy.constraints")
    except ModuleNotFoundError:
        return _Root2DConstraint(frame_indices, root_2d)
    return constraints_module.Root2DConstraintSet(skeleton, frame_indices, root_2d)


def native_constraint_source() -> str:
    """Describe which native-constraint implementation is active."""
    try:
        importlib.import_module("ardy.constraints")
    except ModuleNotFoundError:
        return "production compatibility shim"
    return "ardy.constraints"


def _end_effector_constraint(
    skeleton,
    name: str,
    frame_indices: torch.Tensor,
    positions: torch.Tensor,
    rotations: torch.Tensor,
):
    try:
        constraints_module = importlib.import_module("ardy.constraints")
    except ModuleNotFoundError:
        return _EndEffectorConstraint(
            skeleton,
            frame_indices,
            positions,
            rotations,
            _base_end_effector_name(name),
        )
    constraint_class = {
        "left_hand": constraints_module.LeftHandConstraintSet,
        "right_hand": constraints_module.RightHandConstraintSet,
        "left_foot": constraints_module.LeftFootConstraintSet,
        "right_foot": constraints_module.RightFootConstraintSet,
    }[name]
    return constraint_class(
        skeleton,
        frame_indices=frame_indices,
        global_joints_positions=positions,
        global_joints_rots=rotations,
        root_2d=None,
    )


class _Root2DConstraint:
    def __init__(self, frame_indices: torch.Tensor, root_2d: torch.Tensor) -> None:
        self.frame_indices = frame_indices
        self.root_2d = root_2d

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        data_dict["root_2d"].append(self.root_2d)
        index_dict["root_2d"].append(self.frame_indices)


class _EndEffectorConstraint:
    """Compatibility implementation of NVIDIA's EE constraint semantics."""

    def __init__(
        self,
        skeleton,
        frame_indices: torch.Tensor,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        base_name: str,
    ) -> None:
        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.positions = positions
        self.rotations = rotations
        rot_names, pos_names = skeleton.expand_joint_names([base_name, "Hips"])
        self.rot_indices = torch.tensor(
            [skeleton.bone_order_names.index(name) for name in rot_names],
            device=positions.device,
        )
        self.pos_indices = torch.tensor(
            [skeleton.bone_order_names.index(name) for name in pos_names],
            device=positions.device,
        )
        heading = compute_heading_angle(positions, skeleton)
        self.global_root_heading = torch.stack(
            (torch.cos(heading), torch.sin(heading)), dim=-1
        )

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        data_dict["global_joints_positions"].append(
            self.positions[0, self.pos_indices]
        )
        index_dict["global_joints_positions"].append(
            _constraint_pairs(self.frame_indices, self.pos_indices)
        )
        data_dict["global_joints_rots"].append(self.rotations[0, self.rot_indices])
        index_dict["global_joints_rots"].append(
            _constraint_pairs(self.frame_indices, self.rot_indices)
        )
        root = self.positions[:, self.skeleton.root_idx]
        data_dict["root_2d"].append(root[:, [0, 2]])
        index_dict["root_2d"].append(self.frame_indices)
        data_dict["root_y_pos"].append(root[:, 1])
        index_dict["root_y_pos"].append(self.frame_indices)
        data_dict["global_root_heading"].append(self.global_root_heading)
        index_dict["global_root_heading"].append(self.frame_indices)


def _constraint_pairs(frames: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    return torch.stack((frames.expand(len(joints)), joints), dim=-1)


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
