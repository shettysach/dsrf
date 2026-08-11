from __future__ import annotations

import numpy as np

from motion_gen.kinematic_planner.parser import (
    planner_direction,
    planner_mode,
)
from shared.geometry import local_xy_to_world, yaw_from_quat_wxyz

PLANNER_CONTEXT_FRAMES = 4


def build_planner_inputs(
    context: np.ndarray,
    motion: str,
    target_xy: tuple[float, float] | None,
    direction: str | None = None,
) -> dict[str, np.ndarray]:
    """Build kinematic-planner ONNX inputs from robot-local navigation controls."""
    mode = planner_mode(motion)
    if motion == "stand" and (target_xy is not None or direction is not None):
        raise ValueError("stand requires no target")
    if motion == "walk" and (target_xy is None) == (direction is None):
        raise ValueError("walk requires exactly one target")

    root = context[0, -1]
    root_position = root[:3].astype(np.float32)
    yaw = yaw_from_quat_wxyz(root[3:7])
    facing = _planar_vector(1.0, 0.0, yaw)
    movement = np.zeros(3, dtype=np.float32)
    has_target = np.zeros((1, 1), dtype=np.int64)
    positions = np.zeros((1, PLANNER_CONTEXT_FRAMES, 3), dtype=np.float32)
    headings = np.zeros((1, PLANNER_CONTEXT_FRAMES), dtype=np.float32)

    if direction is not None:
        local_forward, local_left = planner_direction(direction)
        movement = _planar_vector(local_forward, local_left, yaw)
    elif target_xy is not None:
        forward, left = target_xy
        world_delta = _planar_vector(forward, left, yaw)
        distance = float(np.linalg.norm(world_delta[:2]))
        if distance <= 1e-6:
            raise ValueError("walk target_xy must be non-zero")
        movement = world_delta / distance
        positions[:] = root_position + world_delta
        headings[:] = yaw
        has_target[:] = 1

    return {
        "context_mujoco_qpos": context,
        "target_vel": np.array([-1.0], dtype=np.float32),
        "mode": np.array([mode], dtype=np.int64),
        "movement_direction": movement[None],
        "facing_direction": facing[None],
        "random_seed": np.array([1234], dtype=np.int64),
        "has_specific_target": has_target,
        "specific_target_positions": positions,
        "specific_target_headings": headings,
        # Allow the kinematic planner to select its learned 6-16 token horizon.
        "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
        "height": np.array([-1.0], dtype=np.float32),
    }


def _planar_vector(forward: float, left: float, yaw: float) -> np.ndarray:
    vector = np.zeros(3, dtype=np.float32)
    vector[:2] = local_xy_to_world(forward, left, yaw)
    return vector
