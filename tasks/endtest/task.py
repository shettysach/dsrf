from __future__ import annotations

from tasks.spec import SceneSpecFn, TaskSpec, ViewerSpec


def _make_scene() -> SceneSpecFn:
    from tasks.endtest.scene import make_endtest_spec_fn

    return make_endtest_spec_fn()


TASK = TaskSpec(
    name="endtest",
    objective="Place each foot on its matching green target square.",
    make_scene=_make_scene,
    viewer=ViewerSpec(
        distance=2.5,
        elevation=-90.0,
        origin="world",
        lookat=(0.35, 0.0, 0.0),
    ),
)
