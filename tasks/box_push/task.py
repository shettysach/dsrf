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
    # The box near face is x=1.30 m. Start the G1 0.45 m behind it so this
    # first test isolates hand targeting/contact rather than navigation.
    robot_initial_pos=(0.85, 0.0, 0.76),
    observation_camera=ObservationCameraSpec(
        # Head-height, forward-facing view used for the VLM observation stream.
        egocentric=True,
        fovy=90.0,
    ),
)
