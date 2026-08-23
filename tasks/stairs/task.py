from __future__ import annotations

from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec


def _make_scene() -> SceneSpecFn:
    from tasks.stairs.scene import make_stairs_spec_fn

    return make_stairs_spec_fn()


TASK = TaskSpec(
    name="stairs",
    objective="Walk to the smaller stair block without colliding with either block.",
    make_scene=_make_scene,
    observation_camera=ObservationCameraSpec(
        distance=4.0,
        elevation=-50.0,
    ),
)
