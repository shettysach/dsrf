from __future__ import annotations

from tasks.spec import SceneSpecFn, TaskSpec, ViewerSpec


def _make_scene() -> SceneSpecFn:
    from tasks.seesaw.scene import make_seesaw_spec_fn

    return make_seesaw_spec_fn()


TASK = TaskSpec(
    name="seesaw",
    objective="Walk onto the counterweighted seesaw and stand where it balances.",
    make_scene=_make_scene,
    viewer=ViewerSpec(
        distance=4.0,
        elevation=-50.0,
        origin="world",
        lookat=(1.5, 0.0, 0.0),
    ),
)
