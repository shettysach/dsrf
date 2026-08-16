from __future__ import annotations

from tasks.spec import SceneSpecFn, TaskSpec, ViewerSpec


def _make_scene(*, goal_index: int | None = None) -> SceneSpecFn:
    from tasks.sokoban.scene import make_sokoban_spec_fn

    return make_sokoban_spec_fn(variant_index=goal_index or 0)


TASK = TaskSpec(
    name="sokoban",
    objective="Push the yellow box onto the marked green goal region.",
    make_scene=_make_scene,
    viewer=ViewerSpec(distance=6.5, elevation=-50.0),
)
