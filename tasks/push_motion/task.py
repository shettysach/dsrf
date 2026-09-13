from __future__ import annotations

from typing import TYPE_CHECKING

from tasks.push_motion.settings import PushMotionSettings
from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg


def _make_entities() -> dict[str, "EntityCfg"]:
    from tasks.box_push.scene import make_box_push_entity_cfg

    settings = PushMotionSettings.from_env()
    return {
        "box": make_box_push_entity_cfg(
            box_x=settings.box_start_x, box_mass=settings.box_mass
        )
    }


def _make_scene() -> SceneSpecFn:
    from tasks.push_motion.scene import make_push_motion_spec_fn

    return make_push_motion_spec_fn()


_SETTINGS = PushMotionSettings.from_env()


TASK = TaskSpec(
    name="push_motion",
    objective="Reach the box with both palms and push it to the green goal.",
    make_scene=_make_scene,
    make_entities=_make_entities,
    virtual_force_objects=("box",),
    virtual_force_magnitude=_SETTINGS.vf_magnitude,
    robot_initial_pos=(0.0, 0.0, 0.76),
    observation_camera=ObservationCameraSpec(
        world_position=(0.0, -6.5, 4.5),
        world_lookat=(_SETTINGS.goal_x / 2.0, 0.0, 0.8),
        fovy=65.0,
    ),
)
