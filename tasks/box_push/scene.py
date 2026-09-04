from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg

from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]


# A low-friction, ballasted box should slide under a two-palm push instead of
# tipping around its leading edge.
BOX_HALF_SIZE = (0.25, 0.45, 0.50)
BOX_MASS = 3.0
# Put the near face 0.40 m in front of the robot root. The hands make contact
# early enough to keep pushing before the single ARDY window ends.
BOX_START = (1.40, 0.0)
DEFAULT_GOAL_X = 2.05
GOAL_HALF_SIZE = (0.30, 0.50, 0.01)


def _goal_center() -> tuple[float, float]:
    """Read the optional per-workflow goal position."""

    goal_x = float(os.environ.get("BOX_PUSH_GOAL_X", str(DEFAULT_GOAL_X)))
    if not math.isfinite(goal_x):
        raise ValueError("BOX_PUSH_GOAL_X must be finite")
    return (goal_x, 0.0)

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
        init_state=EntityCfg.InitialStateCfg(pos=(*BOX_START, BOX_HALF_SIZE[2])),
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
        # Slightly larger than the box footprint. At the initial pose there is
        # a visible 0.10 m gap between the box and this marker.
        size=GOAL_HALF_SIZE,
        rgba=(0.1, 0.8, 0.2, 0.5),
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
