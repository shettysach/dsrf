from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

type SceneSpecFn = Callable[["MjSpec"], None]
type SceneFactory = Callable[[], SceneSpecFn]
type EntityFactory = Callable[[], dict[str, "EntityCfg"]]


@dataclass(frozen=True)
class ObservationCameraSpec:
    distance: float = 2.0
    elevation: float = -15.0
    azimuth: float = 0.0
    fovy: float = 45.0
    egocentric: bool = False

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
    make_entities: EntityFactory = lambda: {}
    virtual_force_objects: tuple[str, ...] = ()
    virtual_force_magnitude: float = 15.0
    virtual_force_max: float = 30.0
    observation_camera: ObservationCameraSpec = ObservationCameraSpec()
    robot_initial_pos: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("Task name must be non-empty and trimmed")
        if not self.objective.strip():
            raise ValueError("Task objective must be non-empty")
        if any(not name or name != name.strip() for name in self.virtual_force_objects):
            raise ValueError("Virtual-force object names must be non-empty and trimmed")
        if len(set(self.virtual_force_objects)) != len(self.virtual_force_objects):
            raise ValueError("Virtual-force object names must be unique")
        if self.virtual_force_magnitude < 0.0:
            raise ValueError("Virtual-force magnitude must be non-negative")
        if self.virtual_force_max < 0.0:
            raise ValueError("Virtual-force maximum must be non-negative")
        if self.robot_initial_pos is not None and len(self.robot_initial_pos) != 3:
            raise ValueError("Robot initial position must contain exactly three values")
