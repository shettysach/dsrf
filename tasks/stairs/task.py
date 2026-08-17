from __future__ import annotations

from typing import TYPE_CHECKING

from tasks.spec import SceneSpecFn, TaskSpec, TaskStepHook, ViewerSpec

if TYPE_CHECKING:
    from sim.env import MjlabEnv


def _make_scene() -> SceneSpecFn:
    from tasks.stairs.scene import make_stairs_spec_fn

    return make_stairs_spec_fn()


def _make_step_hook(simulation: "MjlabEnv") -> "TaskStepHook":
    from tasks.stairs.grasp import StairsGrasp

    return StairsGrasp(simulation)


TASK = TaskSpec(
    name="stairs",
    objective="Approach the smaller stair block and pick it up with both hands.",
    make_scene=_make_scene,
    viewer=ViewerSpec(distance=4.0, elevation=-50.0, origin="robot"),
    make_step_hook=_make_step_hook,
)
