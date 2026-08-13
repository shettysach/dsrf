from __future__ import annotations

from typing import cast

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.geometry import axis_angle_to_matrix, quaternion_to_matrix


def qpos_to_ardy_inputs(
    qpos: np.ndarray,
    converter: MujocoQposConverter,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MuJoCo G1 qpos into ARDY local rotations and root positions."""
    qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=device)
    if qpos_tensor.ndim != 2 or qpos_tensor.shape[1] != 36:
        raise ValueError(
            f"Initial qpos must have shape [T, 36], got {qpos_tensor.shape}"
        )

    frames = qpos_tensor.shape[0]
    joints = converter.skeleton.nbjoints
    local_rot_mats = (
        torch.eye(3, dtype=torch.float32, device=device)
        .expand(frames, joints, 3, 3)
        .clone()
    )

    ardy_joint_indices = converter._mujoco_indices_to_ardy_indices.to(
        device=device, dtype=torch.long
    )
    joint_axes = converter._mujoco_joint_axis_values_ardy_space.to(device)
    adjusted_joint_rots = axis_angle_to_matrix(
        qpos_tensor[:, 7:, None] * joint_axes[None]
    )
    rotation_offsets = converter._rot_offsets_f2q.to(device)
    local_rot_mats[:, ardy_joint_indices] = torch.matmul(
        rotation_offsets[ardy_joint_indices].transpose(-2, -1)[None],
        adjusted_joint_rots,
    )

    root_index = cast(int, converter.skeleton.root_idx)
    mujoco_root_rot = quaternion_to_matrix(qpos_tensor[:, 3:7])
    mujoco_to_ardy = converter.mujoco_to_ardy_matrix.to(device)
    ardy_to_mujoco = converter.ardy_to_mujoco_matrix.to(device)
    adjusted_root_rot = torch.matmul(
        torch.matmul(mujoco_to_ardy[None], mujoco_root_rot),
        ardy_to_mujoco[None],
    )
    local_rot_mats[:, root_index] = torch.matmul(
        rotation_offsets[root_index].transpose(-2, -1),
        adjusted_root_rot,
    )

    root_positions = torch.matmul(
        mujoco_to_ardy[None], qpos_tensor[:, :3, None]
    ).squeeze(-1)
    return local_rot_mats[None], root_positions[None]


def build_initial_history(
    qpos: np.ndarray,
    converter: MujocoQposConverter,
    motion_rep,
    *,
    device: torch.device,
) -> torch.Tensor:
    local_rot_mats, root_positions = qpos_to_ardy_inputs(
        qpos,
        converter,
        device=device,
    )
    return motion_rep(
        local_rot_mats,
        root_positions,
        to_normalize=True,
    )
