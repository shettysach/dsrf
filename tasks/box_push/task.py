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
    # Enough assistance to overcome the ballasted box's sliding friction while
    # remaining well below its approximate forward-tipping threshold.
    virtual_force_magnitude=4.0,
    virtual_force_max=8.0,
    # The robot starts at the world origin and walks into the contact pose.
    robot_initial_pos=(0.0, 0.0, 0.76),
    observation_camera=ObservationCameraSpec(
        # Head-height, forward-facing view used for the VLM observation stream.
        egocentric=True,
        fovy=90.0,
    ),
)
