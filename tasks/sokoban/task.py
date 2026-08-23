from __future__ import annotations

from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec


def _make_scene() -> SceneSpecFn:
    from tasks.sokoban.scene import make_sokoban_spec_fn

    return make_sokoban_spec_fn()


TASK = TaskSpec(
    name="sokoban",
    objective="Push both boxes onto the two marked goal regions.",
    make_scene=_make_scene,
    observation_camera=ObservationCameraSpec(distance=5.0, elevation=-50.0),
)
