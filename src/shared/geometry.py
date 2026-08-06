"""Small NumPy geometry helpers shared across pipeline processes.

MJLab provides equivalent quaternion and frame-transform helpers, but those operate on
Torch tensors. Keeping these operations in NumPy avoids making the lightweight agent
process import Torch/MJLab and avoids NumPy-to-Torch-to-NumPy conversions when building
Arrow observations and ONNX inputs.
"""

from __future__ import annotations

import math

import numpy as np


def yaw_from_quat_wxyz(quaternion: np.ndarray) -> float:
    """Return world Z yaw from a quaternion stored in MuJoCo's wxyz order."""
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Quaternion must be finite with shape [4]")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        raise ValueError("Quaternion norm must be positive")
    w, x, y, z = quat / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def local_xy_to_world(forward: float, left: float, yaw: float) -> np.ndarray:
    """Rotate robot-local ``(forward, left)`` into world ``(x, y)``."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        [cosine * forward - sine * left, sine * forward + cosine * left],
        dtype=np.float64,
    )


def world_xy_to_local(delta_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a world ``(x, y)`` displacement into local ``(forward, left)``."""
    delta = np.asarray(delta_xy, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("World XY displacement must be finite with shape [2]")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        [
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
        ],
        dtype=np.float64,
    )
