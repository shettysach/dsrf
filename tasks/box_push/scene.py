from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg

from tasks.box_push.settings import BoxPushSettings
from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]


# A lightweight planar cube keeps this scripted contact benchmark aligned with
# the Sokoban object model: it translates on the floor but cannot tip or roll.
_BOX_RGBA = (0.65, 0.42, 0.2, 1.0)


def make_box_push_spec_fn() -> SceneSpecFn:
    """Create the non-colliding goal marker."""

    def add_box_push(spec: MjSpec) -> None:
        _add_goal(spec)

    return add_box_push


def make_box_push_entity_cfg() -> EntityCfg:
    """Create the MJLab-managed dynamic box entity."""
    settings = BoxPushSettings.from_env()

    return EntityCfg(
        spec_fn=_make_box_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(settings.box_x, 0.0, settings.half_size[2])
        ),
    )


def _make_box_spec() -> "MjSpec":
    settings = BoxPushSettings()
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    body = spec.worldbody.add_body(name="box")
    body.add_joint(
        name="box_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,  # ty: ignore[unresolved-attribute]
        axis=(1.0, 0.0, 0.0),
        damping=settings.box_slide_damping,
    )
    body.add_joint(
        name="box_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,  # ty: ignore[unresolved-attribute]
        axis=(0.0, 1.0, 0.0),
        damping=settings.box_slide_damping,
    )
    body.add_geom(
        name="box_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,  # ty: ignore[unresolved-attribute]
        size=settings.half_size,
        rgba=_BOX_RGBA,
        contype=1,
        conaffinity=1,
    )
    return spec


def _add_goal(spec: "MjSpec") -> None:
    settings = BoxPushSettings.from_env()
    spec.worldbody.add_geom(
        name="box_goal",
        type=mujoco.mjtGeom.mjGEOM_BOX,  # ty: ignore[unresolved-attribute]
        pos=(settings.goal_x, 0.0, 0.01),
        # Clearance around the complete box footprint.
        size=(settings.goal_half_size, settings.goal_half_size, 0.01),
        rgba=(0.1, 0.8, 0.2, 0.5),
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
