from __future__ import annotations

from tasks.spec import SceneSpecFn, TaskSpec, ViewerSpec


def _make_scene(*, goal_index: int | None = None) -> SceneSpecFn:
    del goal_index  # Only portrait-corridors uses deterministic goal selection.
    from tasks.sokoban.scene import make_sokoban_spec_fn

    return make_sokoban_spec_fn()


TASK = TaskSpec(
    name="sokoban",
    objective="Push both boxes onto the two marked goal regions.",
    make_scene=_make_scene,
    viewer=ViewerSpec(distance=6.5, elevation=-50.0),
)
