from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from sim.camera import ProjectionContext

MAX_TARGET_DISTANCE_M = 1.5
MIN_TARGET_DISTANCE_M = 0.05
GROUND_Z_M = 0.0
GROUND_TOLERANCE_M = 0.25


@dataclass(frozen=True)
class ResolvedWaypoint:
    normalized: tuple[int, int]
    pixel: tuple[int, int]
    depth: float
    target_xy: tuple[float, float]


@dataclass(frozen=True)
class ResolvedEndEffector:
    name: str
    normalized: tuple[int, int]
    pixel: tuple[int, int]
    depth: float
    target_xyz: tuple[float, float, float]


def resolve_waypoint(
    waypoint: tuple[int, int],
    projection: ProjectionContext,
    *,
    patch_size: int = 5,
    max_distance: float = MAX_TARGET_DISTANCE_M,
) -> ResolvedWaypoint:
    pixel, depth, world_point = _unproject(waypoint, projection, patch_size)
    if abs(float(world_point[2]) - GROUND_Z_M) > GROUND_TOLERANCE_M:
        raise ValueError(
            f"Selected point is not on the floor: world_z={world_point[2]:.3f}"
        )

    target = _world_to_robot(world_point - projection.root_pos_w, projection)
    distance = float(torch.linalg.vector_norm(target[:2]))
    if not math.isfinite(distance):
        raise ValueError("Local waypoint is not finite")
    if distance < MIN_TARGET_DISTANCE_M:
        raise ValueError("Selected waypoint is too close to the robot")
    if distance > max_distance:
        target = target * (max_distance / distance)

    return ResolvedWaypoint(
        normalized=waypoint,
        pixel=pixel,
        depth=depth,
        target_xy=(float(target[0]), float(target[1])),
    )


def resolve_end_effector(
    name: str,
    target_2d: tuple[int, int],
    projection: ProjectionContext,
    *,
    patch_size: int = 5,
) -> ResolvedEndEffector:
    pixel, depth, world_point = _unproject(target_2d, projection, patch_size)
    target_xyz = _world_to_robot(world_point - projection.root_pos_w, projection)
    if not bool(torch.isfinite(target_xyz).all()):
        raise ValueError("Local end-effector target is not finite")
    return ResolvedEndEffector(
        name=name,
        normalized=target_2d,
        pixel=pixel,
        depth=depth,
        target_xyz=(
            float(target_xyz[0]),
            float(target_xyz[1]),
            float(target_xyz[2]),
        ),
    )


def _unproject(
    target_2d: tuple[int, int],
    projection: ProjectionContext,
    patch_size: int,
) -> tuple[tuple[int, int], float, torch.Tensor]:
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("Depth patch size must be a positive odd integer")
    height, width = projection.depth.shape
    x, y = target_2d
    u = round(x / 1000 * (width - 1))
    v = round(y / 1000 * (height - 1))

    radius = patch_size // 2
    patch = projection.depth[
        max(0, v - radius) : min(height, v + radius + 1),
        max(0, u - radius) : min(width, u + radius + 1),
    ]
    valid = patch[
        torch.isfinite(patch)
        & (patch > projection.near)
        & (patch < projection.far * (1.0 - 1e-6))
    ]
    if valid.numel() == 0:
        raise ValueError(f"No valid depth near pixel ({u},{v})")
    depth = float(torch.median(valid))

    right = projection.camera_rotation_w[:, 0]
    up = projection.camera_rotation_w[:, 1]
    forward = -projection.camera_rotation_w[:, 2]
    half_height = math.tan(projection.fovy_rad * 0.5)
    image_x = (2.0 * (u + 0.5) / width - 1.0) * (width / height)
    image_y = 1.0 - 2.0 * (v + 0.5) / height
    world_point = (
        projection.camera_pos_w
        + forward * depth
        + right * (image_x * half_height * depth)
        + up * (image_y * half_height * depth)
    )
    if not bool(torch.isfinite(world_point).all()):
        raise ValueError("Unprojected target is not finite")
    return (u, v), depth, world_point


def _world_to_robot(
    delta_w: torch.Tensor,
    projection: ProjectionContext,
) -> torch.Tensor:
    return quat_apply_inverse(yaw_quat(projection.root_quat_w), delta_w)
