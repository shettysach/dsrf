from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg

from tasks.box_push.settings import BoxPushSettings
from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]


# A low-friction, ballasted box should slide under a two-palm push instead of
# tipping around its leading edge.
BOX_HALF_SIZE = BoxPushSettings().half_size
BOX_MASS = 3.0
BOX_START = (3.0, 0.0)
DEFAULT_GOAL_X = 6.0
GOAL_HALF_SIZE = (0.65, 0.65, 0.01)


def _goal_center() -> tuple[float, float]:
    """Read the optional per-workflow goal position."""

    return (BoxPushSettings.from_env().goal_x, 0.0)


_BOX_RGBA = (0.65, 0.42, 0.2, 1.0)


def make_box_push_spec_fn() -> SceneSpecFn:
    """Create the non-colliding goal marker."""

    def add_box_push(spec: MjSpec) -> None:
        _add_goal(spec)

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
    body.add_freejoint(name="box_free_joint")
    body.add_geom(
        name="box_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,  # ty: ignore[unresolved-attribute]
        size=BOX_HALF_SIZE,
        mass=BOX_MASS,
        friction=(0.2, 0.01, 0.001),
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
