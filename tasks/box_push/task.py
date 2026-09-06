from __future__ import annotations

from typing import TYPE_CHECKING

from tasks.box_push.settings import BoxPushSettings
from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg


def _make_entities() -> dict[str, "EntityCfg"]:
    from tasks.box_push.scene import make_box_push_entity_cfg

    return {"box": make_box_push_entity_cfg()}


def _make_scene() -> SceneSpecFn:
    from tasks.box_push.scene import make_box_push_spec_fn

    return make_box_push_spec_fn()


_SETTINGS = BoxPushSettings.from_env()


TASK = TaskSpec(
    name="box_push",
    objective="Push the box onto the green goal.",
    make_scene=_make_scene,
    make_entities=_make_entities,
    # Contact-gated assistance follows the generated palm motion. It reduces
    # the box resistance only while a palm is physically touching it.
    virtual_force_objects=("box",),
    virtual_force_magnitude=_SETTINGS.virtual_force_magnitude,
    virtual_force_max=_SETTINGS.virtual_force_max,
    # The robot starts at the world origin and walks into the contact pose.
    robot_initial_pos=(0.0, 0.0, 0.76),
    observation_camera=ObservationCameraSpec(
        world_position=(0.0, -6.0, 5.0),
        world_lookat=(_SETTINGS.box_x, 0.0, _SETTINGS.half_size[2]),
        fovy=65.0,
    ),
)
