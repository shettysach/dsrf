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
DEFAULT_SETTINGS = BoxPushSettings()
BOX_HALF_SIZE = DEFAULT_SETTINGS.half_size
BOX_MASS = DEFAULT_SETTINGS.box_mass
BOX_START = (DEFAULT_SETTINGS.box_x, 0.0)
DEFAULT_GOAL_X = DEFAULT_SETTINGS.goal_x
GOAL_HALF_SIZE = (
    DEFAULT_SETTINGS.goal_half_size,
    DEFAULT_SETTINGS.goal_half_size,
    0.01,
)


def _goal_center() -> tuple[float, float]:
    """Read the optional per-workflow goal position."""

    return (BoxPushSettings.from_env().goal_x, 0.0)


_BOX_RGBA = (0.65, 0.42, 0.2, 1.0)
_WELD_SOLREF = DEFAULT_SETTINGS.weld_solref
_WELD_DATA = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def make_box_push_spec_fn() -> SceneSpecFn:
    """Create the non-colliding goal marker."""

    def add_box_push(spec: MjSpec) -> None:
        _add_goal(spec)
        _add_inactive_hand_welds(spec)

    return add_box_push


def make_box_push_entity_cfg() -> EntityCfg:
    """Create the MJLab-managed dynamic box entity."""

    return EntityCfg(
        spec_fn=_make_box_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(BoxPushSettings.from_env().box_x, 0.0, BOX_HALF_SIZE[2])
        ),
    )


def _make_box_spec() -> "MjSpec":
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    body = spec.worldbody.add_body(name="box")
    body.add_joint(
        name="box_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,  # ty: ignore[unresolved-attribute]
        axis=(1.0, 0.0, 0.0),
        damping=DEFAULT_SETTINGS.box_slide_damping,
    )
    body.add_joint(
        name="box_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,  # ty: ignore[unresolved-attribute]
        axis=(0.0, 1.0, 0.0),
        damping=DEFAULT_SETTINGS.box_slide_damping,
    )
    body.add_geom(
        name="box_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,  # ty: ignore[unresolved-attribute]
        size=BOX_HALF_SIZE,
        mass=BOX_MASS,
        friction=DEFAULT_SETTINGS.box_friction,
        rgba=_BOX_RGBA,
        contype=1,
        conaffinity=1,
    )
    return spec


def _add_goal(spec: "MjSpec") -> None:
    spec.worldbody.add_geom(
        name="box_goal",
        type=mujoco.mjtGeom.mjGEOM_BOX,  # ty: ignore[unresolved-attribute]
        pos=(*_goal_center(), 0.01),
        # Clearance around the complete box footprint.
        size=GOAL_HALF_SIZE,
        rgba=(0.1, 0.8, 0.2, 0.5),
        contype=0,
        conaffinity=0,
        mass=0.0,
    )


def _add_inactive_hand_welds(spec: "MjSpec") -> None:
    """Predeclare topology; runtime captures the actual attachment pose."""
    for hand, body in (
        ("left_hand", "robot/left_wrist_yaw_link"),
        ("right_hand", "robot/right_wrist_yaw_link"),
    ):
        spec.add_equality(
            name=f"{hand}_box_weld",
            type=mujoco.mjtEq.mjEQ_WELD,  # ty: ignore[unresolved-attribute]
            objtype=mujoco.mjtObj.mjOBJ_BODY,  # ty: ignore[unresolved-attribute]
            name1=body,
            name2="box/box",
            active=0,
            data=_WELD_DATA,
            solref=_WELD_SOLREF,
        )
