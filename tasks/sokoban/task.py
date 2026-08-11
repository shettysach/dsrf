from __future__ import annotations

from pathlib import Path

from tasks.spec import SceneSpecFn, TaskSpec, ViewerSpec


def _make_scene() -> SceneSpecFn:
    from tasks.sokoban.scene import make_sokoban_spec_fn

    return make_sokoban_spec_fn()


TASK = TaskSpec(
    name="sokoban",
    objective="Push both boxes onto the two marked goal regions.",
    instructions_path=Path(__file__).with_name("TASK.md"),
    make_scene=_make_scene,
    viewer=ViewerSpec(distance=5.0, elevation=-50.0),
)
