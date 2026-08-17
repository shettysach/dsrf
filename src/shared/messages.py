from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOTION_COLUMNS = 36
SONIC_FPS = 50
ARDY_EMBEDDING_SIZE = 4096
END_EFFECTOR_NAMES = frozenset({"left_hand", "right_hand", "left_foot", "right_foot"})


@dataclass(frozen=True)
class EndEffectorSelection:
    name: str
    target_2d: tuple[int, int]

    def __post_init__(self) -> None:
        if self.name not in END_EFFECTOR_NAMES:
            raise ValueError(f"Unsupported end effector: {self.name}")
        if len(self.target_2d) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.target_2d
        ):
            raise ValueError("End-effector image target must contain two integers")
        if not all(0 <= value <= 1000 for value in self.target_2d):
            raise ValueError("End-effector image coordinates must be in [0,1000]")


@dataclass(frozen=True)
class EndEffectorTarget:
    name: str
    target_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.name not in END_EFFECTOR_NAMES:
            raise ValueError(f"Unsupported end effector: {self.name}")
        if not all(np.isfinite(value) for value in self.target_xyz):
            raise ValueError("End-effector target must be finite")


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
    target_xys: tuple[tuple[float, float], ...]
    direction: str | None = None
    end_effectors: tuple[EndEffectorTarget, ...] = ()
    reasoning: str | None = None

    def __post_init__(self) -> None:
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Command is empty")
        _validate_navigation(self.motion, self.target_xys, self.direction)
        _validate_end_effectors(self.end_effectors)
        object.__setattr__(self, "text", normalized)


@dataclass(frozen=True)
class VisualObservation:
    observation_id: int
    completed_command: str | None
    jpeg: bytes
    run_id: int = 0
    collision_detected: bool = False

    def __post_init__(self) -> None:
        if not self.jpeg:
            raise ValueError("Observation JPEG is empty")


@dataclass(frozen=True)
class GroundingRequest:
    observation_id: int
    waypoints_2d: tuple[tuple[int, int], ...]
    end_effectors_2d: tuple[EndEffectorSelection, ...] = ()

    def __post_init__(self) -> None:
        for x, y in self.waypoints_2d:
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (x, y)
            ):
                raise ValueError("Waypoint coordinates must be integers")
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                raise ValueError("Waypoint coordinates must be in [0,1000]")
        names = [selection.name for selection in self.end_effectors_2d]
        if len(names) != len(set(names)):
            raise ValueError("Each end effector may be grounded once")


@dataclass(frozen=True)
class GroundingResult:
    observation_id: int
    target_xys: tuple[tuple[float, float], ...]
    end_effectors: tuple[EndEffectorTarget, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(value) for target_xy in self.target_xys for value in target_xy
        ):
            raise ValueError("target_xys must be finite")
        _validate_end_effectors(self.end_effectors)


@dataclass(frozen=True)
class PipelineError:
    source: str
    observation_id: int
    detail: str

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Error source is empty")
        if not self.detail:
            raise ValueError("Error detail is empty")


def _validate_navigation(
    motion: str, target_xys: tuple[tuple[float, float], ...], direction: str | None
) -> None:
    if not motion.strip():
        raise ValueError("Motion prompt must not be empty")
    if target_xys and direction is not None:
        raise ValueError("Motion command cannot have both a waypoint and direction")
    if direction is not None:
        if direction not in {"forward", "backward", "left", "right"}:
            raise ValueError("Unsupported direction")
        return
    if not all(np.isfinite(value) for target_xy in target_xys for value in target_xy):
        raise ValueError("target_xys must be finite")


def _validate_end_effectors(end_effectors: tuple[EndEffectorTarget, ...]) -> None:
    names = [target.name for target in end_effectors]
    if len(names) != len(set(names)):
        raise ValueError("Each end effector may be constrained once")
