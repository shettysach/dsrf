from __future__ import annotations

from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec


def _make_scene() -> SceneSpecFn:
    from tasks.seesaw.scene import make_seesaw_spec_fn

    return make_seesaw_spec_fn()


TASK = TaskSpec(
    name="seesaw",
    objective="Walk onto the counterweighted seesaw and stand where it balances.",
    make_scene=_make_scene,
    observation_camera=ObservationCameraSpec(
        distance=4.0,
        elevation=-50.0,
    ),
)
