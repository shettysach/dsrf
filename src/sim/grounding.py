from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shared.geometry import world_xy_to_local, yaw_from_quat_wxyz
from shared.messages import ProjectionContext

MAX_TARGET_DISTANCE_M = 1.5
MIN_TARGET_DISTANCE_M = 0.05
MAX_HAND_TARGET_DISTANCE_M = 1.25
MAX_FOOT_TARGET_DISTANCE_M = 1.5
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

    delta = world_point - projection.root_pos_w
    target = world_xy_to_local(
        delta[:2],
        yaw_from_quat_wxyz(projection.root_quat_w),
    )
    distance = float(np.linalg.norm(target))
    if not math.isfinite(distance):
        raise ValueError("Local waypoint is not finite")
    if distance < MIN_TARGET_DISTANCE_M:
        raise ValueError("Selected waypoint is too close to the robot")
    if distance > max_distance:
        target *= max_distance / distance

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
    max_distance: float | None = None,
) -> ResolvedEndEffector:
    pixel, depth, world_point = _unproject(target_2d, projection, patch_size)
    delta = world_point - projection.root_pos_w
    target_xy = world_xy_to_local(
        delta[:2],
        yaw_from_quat_wxyz(projection.root_quat_w),
    )
    target_xyz = np.array([target_xy[0], target_xy[1], delta[2]])
    distance = float(np.linalg.norm(target_xyz))
    if not math.isfinite(distance):
        raise ValueError("Local end-effector target is not finite")
    limit = max_distance or _reach_limit(name)
    if distance > limit:
        raise ValueError(
            f"{name} target is out of reach: distance={distance:.3f}m, "
            f"maximum={limit:.3f}m"
        )
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


def _reach_limit(name: str) -> float:
    return (
        MAX_FOOT_TARGET_DISTANCE_M
        if name.endswith("_foot")
        else MAX_HAND_TARGET_DISTANCE_M
    )


def _unproject(
    target_2d: tuple[int, int],
    projection: ProjectionContext,
    patch_size: int,
) -> tuple[tuple[int, int], float, np.ndarray]:
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
        np.isfinite(patch)
        & (patch > projection.near)
        & (patch < projection.far * (1.0 - 1e-6))
    ]
    if not valid.size:
        raise ValueError(f"No valid depth near pixel ({u},{v})")
    depth = float(np.median(valid))

    forward = _unit(projection.camera_forward_w, "camera forward")
    up = _unit(projection.camera_up_w, "camera up")
    right = _unit(np.cross(forward, up), "camera right")
    half_height = projection.frustum_height * 0.5
    image_x = (2.0 * (u + 0.5) / width - 1.0) * (width / height)
    image_y = 1.0 - 2.0 * (v + 0.5) / height
    world_point = (
        projection.camera_pos_w
        + forward * depth
        + right * (image_x * half_height * depth)
        + up * (image_y * half_height * depth)
    )
    if not np.isfinite(world_point).all():
        raise ValueError("Unprojected target is not finite")
    return (u, v), depth, world_point


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"Invalid {name} vector")
    return np.asarray(value, dtype=np.float64) / norm
