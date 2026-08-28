from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg

from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]


# A slightly broader, taller face gives both hands more usable contact area.
BOX_HALF_SIZE = (0.25, 0.35, 0.55)
BOX_MASS = 1.0
BOX_START = (1.75, 0.0)
GOAL_CENTER = (5.0, 0.0)

_BOX_RGBA = (0.45, 0.24, 0.1, 1.0)


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
        friction=(0.75, 0.01, 0.001),
        rgba=_BOX_RGBA,
        contype=1,
        conaffinity=1,
    )
    return spec


def _add_goal(spec: "MjSpec") -> None:
    spec.worldbody.add_geom(
        name="box_goal",
        type=mujoco.mjtGeom.mjGEOM_BOX,  # ty: ignore[unresolved-attribute]
        pos=(*GOAL_CENTER, 0.01),
        size=(0.55, 0.55, 0.01),
        rgba=(0.1, 0.8, 0.2, 0.5),
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
