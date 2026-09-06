import torch
from ardy.constraints import (
    LeftFootConstraintSet,
    LeftHandConstraintSet,
    RightFootConstraintSet,
    RightHandConstraintSet,
    Root2DConstraintSet,
)
from ardy.motion_rep.tools import RotateFeatures

from motion_gen.targets import TimedTargets
from shared.messages import EndEffectorTarget

_JOINT_NAMES = {
    "left_hand": "left_hand_roll_skel",
    "right_hand": "right_hand_roll_skel",
    "left_foot": "left_toe_base",
    "right_foot": "right_toe_base",
}
# ARDY's G1 converter maps MuJoCo's +X (the palm collision capsule's long
# axis) into ARDY's +Z.  Hand targets use semantic local coordinates
# (forward, left, up), so this is the axis that must point into a surface.
_PALM_FORWARD_AXIS_ARDY = torch.tensor((0.0, 0.0, 1.0))
_WRIST_ROTATION_JOINTS = {
    "left_hand": "left_wrist_yaw_skel",
    "right_hand": "right_wrist_yaw_skel",
}
_MAX_END_EFFECTOR_REACH_M = {
    "left_hand": 1.25,
    "right_hand": 1.25,
    "left_foot": 1.5,
    "right_foot": 1.5,
}


def build_timed_constraints(
    motion_rep,
    root_history: torch.Tensor,
    root_heading: torch.Tensor,
    samples: tuple[TimedTargets, ...],
    reference_decoded: dict[str, torch.Tensor] | None = None,
    *,
    generated_frames: int,
    history_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition each sample at its own timestamp, never retime the task goal."""
    indices = [sample.frame for sample in samples]
    if not indices or indices != sorted(set(indices)):
        raise ValueError("Timed targets require unique, increasing frame indices")
    if indices[0] < 0 or indices[-1] >= generated_frames:
        raise ValueError("Timed target lies outside the generated window")
    root = root_history[-1].to(device)
    heading = root_heading.reshape(()).to(device)
    local = torch.tensor(
        [[s.root_xy[1], s.root_xy[0]] for s in samples],
        dtype=root.dtype,
        device=device,
    )
    root_targets_2d = root[[0, 2]] + _rotate_2d(local, heading)
    frames = torch.tensor(indices, device=device) + history_frames
    constraints = [
        _root_constraint(
            motion_rep.skeleton,
            frames,
            root_targets_2d,
            heading.expand(len(samples)),
        )
    ]
    if reference_decoded is not None:
        for sample, frame, root_target_2d in zip(
            samples, frames, root_targets_2d, strict=True
        ):
            if not sample.end_effectors:
                continue
            positions = reference_decoded["posed_joints"][:, sample.frame].to(device)
            rotations = reference_decoded["global_rot_mats"][:, sample.frame].to(device)
            root_pose = root.clone()
            root_pose[[0, 2]] = root_target_2d
            targets = global_end_effector_targets(
                root, heading, sample.end_effectors, device=device
            )
            _validate_end_effector_reach(
                sample.end_effectors,
                targets,
                root_pose,
            )
            if sample.torso_upright:
                constraints.append(
                    _joint_rotation_constraint(
                        motion_rep.skeleton,
                        "waist_pitch_skel",
                        frame.reshape(1),
                        _upright_rotation(heading, rotations).reshape(1, 3, 3),
                    )
                )
            for target, xyz in zip(sample.end_effectors, targets, strict=True):
                edited, edited_rotations = _edit_end_effector_pose(
                    positions,
                    rotations,
                    motion_rep.skeleton,
                    target,
                    xyz,
                    heading,
                    root_position=root_pose if sample.torso_upright else None,
                    upright_root=sample.torso_upright,
                )
                constraints.append(
                    _end_effector_constraint(
                        motion_rep.skeleton,
                        target.name,
                        frame.reshape(1),
                        edited,
                        edited_rotations,
                    )
                )
    observed, mask = motion_rep.create_conditions_from_constraints(
        constraints, history_frames + generated_frames, False, str(device)
    )
    return mask.unsqueeze(0), (motion_rep.normalize(observed) * mask).unsqueeze(0)


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
        constraints.append(
            _root_constraint(motion_rep.skeleton, frame_indices, root_2d)
        )

    if end_effectors:
        if reference_decoded is None:
            raise ValueError("ARDY end-effector constraints require a reference pose")
        target_positions = global_end_effector_targets(
            current_root, heading, end_effectors, device=device
        )
        positions, rotations = _reference_final_pose(reference_decoded, device)
        final_root = current_root.clone()
        if target_xys:
            final_root[[0, 2]] = root_2d[-1]
        _validate_end_effector_reach(end_effectors, target_positions, final_root)
        frame = torch.tensor([generated_frames + history_frames - 1], device=device)
        for target, target_position in zip(
            end_effectors, target_positions, strict=True
        ):
            edited_positions, edited_rotations = _edit_end_effector_pose(
                positions,
                rotations,
                motion_rep.skeleton,
                target,
                target_position,
                heading,
            )
            constraints.append(
                _end_effector_constraint(
                    motion_rep.skeleton,
                    target.name,
                    frame,
                    edited_positions,
                    edited_rotations,
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


def _root_constraint(
    skeleton,
    frame_indices: torch.Tensor,
    root_2d: torch.Tensor,
    heading: torch.Tensor | None = None,
):
    return Root2DConstraintSet(
        skeleton,
        frame_indices,
        root_2d,
        global_root_heading=heading,
    )


def native_constraint_source() -> str:
    """Describe the constraint implementation used by this generator."""
    return "ardy.constraints"


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


class _JointRotationConstraint:
    def __init__(
        self,
        frame_indices: torch.Tensor,
        joint_index: int,
        rotations: torch.Tensor,
    ) -> None:
        self.frame_indices = frame_indices
        self.joint_index = joint_index
        self.rotations = rotations

    def update_constraints(
        self,
        data: dict[str, list[torch.Tensor]],
        indices: dict[str, list[torch.Tensor]],
    ) -> None:
        joint_indices = torch.full_like(self.frame_indices, self.joint_index)
        indices["global_joints_rots"].append(
            torch.stack((self.frame_indices, joint_indices), dim=-1)
        )
        data["global_joints_rots"].append(self.rotations)


def _joint_rotation_constraint(
    skeleton,
    joint_name: str,
    frame_indices: torch.Tensor,
    rotations: torch.Tensor,
) -> _JointRotationConstraint:
    return _JointRotationConstraint(
        frame_indices,
        skeleton.bone_order_names.index(joint_name),
        rotations,
    )


def _edit_end_effector_pose(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    skeleton,
    target: EndEffectorTarget,
    target_position: torch.Tensor,
    heading: torch.Tensor,
    *,
    root_position: torch.Tensor | None = None,
    upright_root: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an FK-consistent hand chain around the requested endpoint."""
    edited_rotations = rotations.clone()
    root_index = skeleton.root_idx
    if upright_root:
        edited_rotations[0, root_index] = _upright_rotation(heading, rotations)
    edited_rotations = _with_palm_normal(
        edited_rotations,
        skeleton,
        target,
        heading,
    )

    endpoint = skeleton.bone_order_names.index(_JOINT_NAMES[target.name])
    _, chain_names = skeleton.expand_joint_names([_base_end_effector_name(target.name)])
    chain = [skeleton.bone_order_names.index(name) for name in chain_names]
    base = chain[0]
    source_rotation = rotations[0, base]
    target_rotation = edited_rotations[0, base]
    endpoint_offsets = positions[0, chain] - positions[0, endpoint]
    local_offsets = endpoint_offsets @ source_rotation

    edited_positions = positions.clone()
    edited_positions[0, chain] = target_position + local_offsets @ target_rotation.T
    if root_position is not None:
        edited_positions[0, root_index] = root_position
    return edited_positions, edited_rotations


def _upright_rotation(
    heading: torch.Tensor, reference_rotations: torch.Tensor
) -> torch.Tensor:
    """Construct an upright ARDY global rotation at the current heading."""
    heading = heading.to(
        device=reference_rotations.device,
        dtype=reference_rotations.dtype,
    )
    cosine, sine = torch.cos(heading), torch.sin(heading)
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    return torch.stack(
        (
            torch.stack((cosine, zero, sine)),
            torch.stack((zero, one, zero)),
            torch.stack((-sine, zero, cosine)),
        )
    )


def _with_palm_normal(
    rotations: torch.Tensor,
    skeleton,
    target: EndEffectorTarget,
    heading: torch.Tensor,
) -> torch.Tensor:
    """Align a hand's palm-forward axis, retaining ARDY's reference twist."""
    if target.palm_normal is None:
        return rotations
    try:
        joint_name = _WRIST_ROTATION_JOINTS[target.name]
    except KeyError:
        return rotations

    local_axis = torch.tensor(
        (target.palm_normal[1], target.palm_normal[2], target.palm_normal[0]),
        dtype=rotations.dtype,
        device=rotations.device,
    )
    desired_axis = _upright_rotation(
        heading, rotations
    ) @ torch.nn.functional.normalize(
        local_axis,
        dim=0,
    )
    joint = skeleton.bone_order_names.index(joint_name)
    current_rotation = rotations[0, joint]
    source_axis = current_rotation @ _PALM_FORWARD_AXIS_ARDY.to(
        device=rotations.device, dtype=rotations.dtype
    )
    alignment = _rotation_between_unit_vectors(source_axis, desired_axis)
    edited = rotations.clone()
    edited[0, joint] = alignment @ current_rotation
    return edited


def _rotation_between_unit_vectors(
    source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return the shortest proper rotation from ``source`` to ``target``."""
    source = torch.nn.functional.normalize(source, dim=0)
    target = torch.nn.functional.normalize(target, dim=0)
    cosine = torch.clamp(torch.dot(source, target), -1.0, 1.0)
    cross = torch.linalg.cross(source, target)
    sine = torch.linalg.vector_norm(cross)
    identity = torch.eye(3, dtype=source.dtype, device=source.device)
    if float(sine) < 1e-6:
        if float(cosine) > 0.0:
            return identity
        basis = torch.tensor((1.0, 0.0, 0.0), dtype=source.dtype, device=source.device)
        if abs(float(source[0])) > 0.9:
            basis = torch.tensor(
                (0.0, 1.0, 0.0), dtype=source.dtype, device=source.device
            )
        axis = torch.nn.functional.normalize(torch.linalg.cross(source, basis), dim=0)
        return 2.0 * torch.outer(axis, axis) - identity
    skew = torch.zeros((3, 3), dtype=source.dtype, device=source.device)
    skew[0, 1], skew[0, 2] = -cross[2], cross[1]
    skew[1, 0], skew[1, 2] = cross[2], -cross[0]
    skew[2, 0], skew[2, 1] = -cross[1], cross[0]
    return identity + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


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
