from __future__ import annotations

from tasks.spec import SceneSpecFn, TaskSpec, ViewerSpec


def _make_scene() -> SceneSpecFn:
    from tasks.stairs.scene import make_stairs_spec_fn

    return make_stairs_spec_fn()


TASK = TaskSpec(
    name="stairs",
    objective="Approach the smaller stair block and pick it up with both hands.",
    make_scene=_make_scene,
    viewer=ViewerSpec(distance=4.0, elevation=-50.0, origin="robot"),
)
