from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

    from sim.env import MjlabEnv

type SceneSpecFn = Callable[["MjSpec"], None]
type SceneFactory = Callable[[], SceneSpecFn]
type SceneFactoryWithGoal = Callable[[int | None], SceneSpecFn]
type ViewerOrigin = Literal["robot", "world"]


class TaskStepHook(Protocol):
    """Task-specific logic run immediately before each physics step."""

    def before_step(self) -> None: ...


type TaskStepHookFactory = Callable[["MjlabEnv"], TaskStepHook]


@dataclass(frozen=True)
class ViewerSpec:
    distance: float = 2.0
    elevation: float = -15.0
    origin: ViewerOrigin = "robot"
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    azimuth: float = 0.0


@dataclass(frozen=True)
class TaskSpec:
    name: str
    objective: str
    make_scene: SceneFactory
    viewer: ViewerSpec = ViewerSpec()
    make_step_hook: TaskStepHookFactory | None = None
    make_scene_with_goal: SceneFactoryWithGoal | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("Task name must be non-empty and trimmed")
        if not self.objective.strip():
            raise ValueError("Task objective must be non-empty")
