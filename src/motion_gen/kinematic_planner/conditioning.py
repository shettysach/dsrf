from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_apply_yaw

from motion_gen.kinematic_planner.parser import (
    planner_direction,
    planner_mode,
)

PLANNER_CONTEXT_FRAMES = 4


def build_planner_inputs(
    context: torch.Tensor,
    motion: str,
    target_xy: tuple[float, float] | torch.Tensor | None,
    direction: str | None = None,
) -> dict[str, torch.Tensor]:
    """Build kinematic-planner ONNX inputs from robot-local navigation controls."""
    mode = planner_mode(motion)
    if motion == "stand" and target_xy is not None:
        raise ValueError("stand cannot use a waypoint target")
    if motion == "walk" and (target_xy is None) == (direction is None):
        raise ValueError("walk requires exactly one target")

    root = context[0, -1]
    root_position = root[:3]
    root_quat = root[3:7]
    facing = _planar_vector(1.0, 0.0, root_quat)
    movement = torch.zeros(3, dtype=context.dtype, device=context.device)
    has_target = torch.zeros((1, 1), dtype=torch.int64, device=context.device)
    positions = torch.zeros(
        (1, PLANNER_CONTEXT_FRAMES, 3),
        dtype=context.dtype,
        device=context.device,
    )
    headings = torch.zeros(
        (1, PLANNER_CONTEXT_FRAMES),
        dtype=context.dtype,
        device=context.device,
    )

    if direction is not None:
        local_forward, local_left = planner_direction(direction)
        movement = _planar_vector(local_forward, local_left, root_quat)
    elif target_xy is not None:
        if isinstance(target_xy, torch.Tensor):
            local_xy = target_xy
        else:
            if math.hypot(*target_xy) <= 1e-6:
                raise ValueError("walk target_xy must be non-zero")
            local_xy = context.new_tensor(target_xy)
        local_delta = torch.cat((local_xy, context.new_zeros(1)))
        world_delta = quat_apply_yaw(root_quat, local_delta)
        movement = world_delta / torch.linalg.vector_norm(local_xy).clamp_min(1e-6)
        positions[:] = root_position + world_delta
        headings[:] = torch.atan2(facing[1], facing[0])
        has_target[:] = 1

    return {
        "context_mujoco_qpos": context,
        "target_vel": context.new_tensor([-1.0]),
        "mode": torch.tensor([mode], dtype=torch.int64, device=context.device),
        "movement_direction": movement[None],
        "facing_direction": facing[None],
        "random_seed": torch.tensor([1234], dtype=torch.int64, device=context.device),
        "has_specific_target": has_target,
        "specific_target_positions": positions,
        "specific_target_headings": headings,
        # Allow the kinematic planner to select its learned 6-16 token horizon.
        "allowed_pred_num_tokens": torch.ones(
            (1, 11), dtype=torch.int64, device=context.device
        ),
        "height": context.new_tensor([-1.0]),
    }


def _planar_vector(
    forward: float,
    left: float,
    root_quat: torch.Tensor,
) -> torch.Tensor:
    local = root_quat.new_tensor((forward, left, 0.0))
    return quat_apply_yaw(root_quat, local)
