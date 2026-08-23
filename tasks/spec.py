from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

type SceneSpecFn = Callable[["MjSpec"], None]
type SceneFactory = Callable[[], SceneSpecFn]


@dataclass(frozen=True)
class ObservationCameraSpec:
    distance: float = 2.0
    elevation: float = -15.0
    azimuth: float = 0.0
    fovy: float = 45.0

    def __post_init__(self) -> None:
        if self.distance <= 0.0:
            raise ValueError("Observation camera distance must be positive")
        if not -89.0 < self.elevation < 89.0:
            raise ValueError("Observation camera elevation must be between -89 and 89")
        if not 0.0 < self.fovy < 180.0:
            raise ValueError("Observation camera FOV must be between 0 and 180")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    objective: str
    make_scene: SceneFactory
    observation_camera: ObservationCameraSpec = ObservationCameraSpec()

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("Task name must be non-empty and trimmed")
        if not self.objective.strip():
            raise ValueError("Task objective must be non-empty")
