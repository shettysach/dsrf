from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOTION_COLUMNS = 36
REFERENCE_HZ = 50
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
class MotionChunk:
    observation_id: int
    command: str
    qpos: np.ndarray

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
        # Arrow exposes received numeric buffers as read-only NumPy views.  Own a
        # writable buffer here because the tracker converts qpos to Torch tensors,
        # whose writable-storage contract cannot be satisfied by those views.
        object.__setattr__(
            self, "qpos", np.array(qpos, dtype=np.float32, order="C", copy=True)
        )


@dataclass(frozen=True)
class AgentCommand:
    observation_id: int
    text: str
    motion: str
    target_xys: tuple[tuple[float, float], ...]
    direction: str | None = None
    end_effectors: tuple[EndEffectorTarget, ...] = ()

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
