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
    # Enough assistance to overcome box friction without launching a 1 kg box.
    virtual_force_magnitude=4.0,
    virtual_force_max=8.0,
    # The box near face is x=1.15 m, 0.40 m ahead of the G1 root.
    robot_initial_pos=(0.75, 0.0, 0.76),
    observation_camera=ObservationCameraSpec(
        # Head-height, forward-facing view used for the VLM observation stream.
        egocentric=True,
        fovy=90.0,
    ),
)
