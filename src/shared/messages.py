from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    # Robot-local (forward, left, up) direction in which a hand palm faces.
    # ``None`` retains the orientation proposed by ARDY's reference motion.
    palm_normal: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.name not in END_EFFECTOR_NAMES:
            raise ValueError(f"Unsupported end effector: {self.name}")
        if not all(np.isfinite(value) for value in self.target_xyz):
            raise ValueError("End-effector target must be finite")
        if self.palm_normal is not None:
            if self.name not in {"left_hand", "right_hand"}:
                raise ValueError("Only hand targets may specify a palm normal")
            if (
                len(self.palm_normal) != 3
                or not all(np.isfinite(value) for value in self.palm_normal)
                or not np.linalg.norm(self.palm_normal) > 1e-6
            ):
                raise ValueError("Palm normal must be a finite, non-zero 3D vector")


@dataclass(frozen=True)
class ContactGoal:
    """Script-grounded maintained contact: body-local points, world XY goal.

    Point ``palm_normal`` values use the same body-local basis as their positions.
    """

    body: str
    target_xy: tuple[float, float]
    points: tuple[EndEffectorTarget, ...]
    goal_half_size: float = 0.65
    maintain_contact: bool = False

    def __post_init__(self) -> None:
        if not self.body or len(self.target_xy) != 2:
            raise ValueError("Contact goal requires a body and world XY destination")
        if not all(np.isfinite(v) for v in (*self.target_xy, self.goal_half_size)):
            raise ValueError("Contact goal must be finite")
        if self.goal_half_size <= 0 or not self.points:
            raise ValueError("Contact goal requires points and a positive goal size")
        _validate_end_effectors(self.points)
        if any(p.name not in {"left_hand", "right_hand"} for p in self.points):
            raise ValueError("Maintained contact currently supports hands only")


@dataclass(frozen=True)
class RootPathGoal:
    """A scripted, contact-free root path with staging and destination points."""

    approach_xy: tuple[float, float]
    target_xy: tuple[float, float]
    points: tuple[EndEffectorTarget, ...] = ()

    def __post_init__(self) -> None:
        if len(self.approach_xy) != 2 or len(self.target_xy) != 2:
            raise ValueError("Root path requires 2D staging and destination points")
        if not all(
            np.isfinite(value) for value in (*self.approach_xy, *self.target_xy)
        ):
            raise ValueError("Root path points must be finite")
        _validate_end_effectors(self.points)
        if self.points and {point.name for point in self.points} != {
            "left_hand",
            "right_hand",
        }:
            raise ValueError("Root path hand constraints require both hands")


@dataclass(frozen=True)
class AgentCommand:
    observation_id: int
    text: str
    motion: str
    target_xys: tuple[tuple[float, float], ...]
    direction: str | None = None
    end_effectors: tuple[EndEffectorTarget, ...] = ()
    reasoning: str | None = None
    terminal: bool = False
    contact_goal: ContactGoal | None = None
    root_path_goal: RootPathGoal | None = None

    def __post_init__(self) -> None:
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Command is empty")
        _validate_navigation(self.motion, self.target_xys, self.direction)
        _validate_end_effectors(self.end_effectors)
        if self.contact_goal is not None and self.root_path_goal is not None:
            raise ValueError("A command cannot contain two scripted goals")
        object.__setattr__(self, "text", normalized)


@dataclass(frozen=True)
class VisualObservation:
    observation_id: int
    completed_command: str | None
    jpeg: bytes
    execution_feedback: str | None = None

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
