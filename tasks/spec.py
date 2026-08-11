from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

type SceneSpecFn = Callable[["MjSpec"], None]
type SceneFactory = Callable[[], SceneSpecFn]


@dataclass(frozen=True)
class ViewerSpec:
    distance: float = 2.0
    elevation: float = -15.0


@dataclass(frozen=True)
class TaskSpec:
    name: str
    objective: str
    make_scene: SceneFactory
    viewer: ViewerSpec = ViewerSpec()

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("Task name must be non-empty and trimmed")
        if not self.objective.strip():
            raise ValueError("Task objective must be non-empty")
