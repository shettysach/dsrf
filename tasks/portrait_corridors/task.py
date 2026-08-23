from __future__ import annotations

from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec


def _make_scene() -> SceneSpecFn:
    from tasks.portrait_corridors.scene import make_portrait_corridors_spec_fn

    return make_portrait_corridors_spec_fn()


TASK = TaskSpec(
    name="portrait-corridors",
    objective="Stand in front of the image of the creator of Linux.",
    make_scene=_make_scene,
    observation_camera=ObservationCameraSpec(distance=3.5, elevation=-30.0),
)
