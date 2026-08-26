from __future__ import annotations

from typing import TYPE_CHECKING

from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg


def _make_entities() -> dict[str, "EntityCfg"]:
    from tasks.box_push.scene import make_box_push_entity_cfg

    return {"box": make_box_push_entity_cfg()}


def _make_scene() -> SceneSpecFn:
    from tasks.box_push.scene import make_box_push_spec_fn

    return make_box_push_spec_fn()


TASK = TaskSpec(
    name="box_push",
    objective="Push the box onto the green goal.",
    make_scene=_make_scene,
    make_entities=_make_entities,
    virtual_force_objects=("box",),
    observation_camera=ObservationCameraSpec(
        distance=4.0,
        elevation=-35.0,
    ),
)
