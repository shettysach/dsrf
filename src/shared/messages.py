from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOTION_COLUMNS = 36
SONIC_FPS = 50
ARDY_EMBEDDING_SIZE = 4096


@dataclass(frozen=True)
class ProjectionContext:
    depth: np.ndarray
    camera_pos_w: np.ndarray
    camera_forward_w: np.ndarray
    camera_up_w: np.ndarray
    frustum_height: float
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    near: float
    far: float

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float32)
        if depth.ndim != 2 or not depth.size:
            raise ValueError(f"Depth must have shape [H, W], got {depth.shape}")
        vectors = {
            "camera_pos_w": (self.camera_pos_w, (3,)),
            "camera_forward_w": (self.camera_forward_w, (3,)),
            "camera_up_w": (self.camera_up_w, (3,)),
            "root_pos_w": (self.root_pos_w, (3,)),
            "root_quat_w": (self.root_quat_w, (4,)),
        }
        for name, (value, shape) in vectors.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, np.ascontiguousarray(array))
        if not np.isfinite(self.frustum_height) or self.frustum_height <= 0.0:
            raise ValueError("Camera frustum height must be positive and finite")
        if not np.isfinite(self.near) or not np.isfinite(self.far):
            raise ValueError("Camera clipping distances must be finite")
        if self.near <= 0.0 or self.far <= self.near:
            raise ValueError("Camera clipping distances are invalid")
        object.__setattr__(self, "depth", np.ascontiguousarray(depth))


@dataclass(frozen=True)
class MotionChunk:
    observation_id: int
    command: str
    qpos: np.ndarray
    reasoning: str | None = None

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if not self.command.strip():
            raise ValueError("Motion command is empty")
        qpos = np.asarray(self.qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != MOTION_COLUMNS:
            raise ValueError(
                f"Motion qpos must have shape [T, {MOTION_COLUMNS}], got {qpos.shape}"
            )
        if qpos.shape[0] == 0:
            raise ValueError("Motion chunk must contain at least one frame")
        if not np.isfinite(qpos).all():
            raise ValueError("Motion chunk contains NaN or infinite values")
        object.__setattr__(self, "qpos", np.ascontiguousarray(qpos))


@dataclass(frozen=True)
class AgentCommand:
    observation_id: int
    text: str
    motion: str
    target_xy: tuple[float, float] | None
    direction: str | None = None
    reasoning: str | None = None

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Command is empty")
        _validate_navigation(self.motion, self.target_xy, self.direction)
        object.__setattr__(self, "text", normalized)


@dataclass(frozen=True)
class VisualObservation:
    observation_id: int
    completed_command: str | None
    jpeg: bytes
    projection: ProjectionContext | None = None
    run_id: int = 0
    collision_detected: bool = False

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if self.run_id < 0:
            raise ValueError("Run ID must be non-negative")
        if not self.jpeg:
            raise ValueError("Observation JPEG is empty")


@dataclass(frozen=True)
class PipelineError:
    source: str
    observation_id: int
    detail: str

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if not self.source:
            raise ValueError("Error source is empty")
        if not self.detail:
            raise ValueError("Error detail is empty")


def _validate_navigation(
    motion: str, target_xy: tuple[float, float] | None, direction: str | None
) -> None:
    if motion not in {"stand", "walk"}:
        raise ValueError("Motion must be stand or walk")
    if motion == "stand":
        if target_xy is not None or direction is not None:
            raise ValueError("Stand command must not have a target")
        return
    if (target_xy is None) == (direction is None):
        raise ValueError("Walk command requires exactly one target")
    if direction is not None:
        if direction not in {"forward", "backward", "left", "right"}:
            raise ValueError("Unsupported direction")
        return
    assert target_xy is not None
    if not all(np.isfinite(value) for value in target_xy):
        raise ValueError("target_xy must be finite")
