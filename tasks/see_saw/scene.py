from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco

from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]
MJ_JOINT_HINGE = mujoco.mjtJoint.mjJNT_HINGE  # ty: ignore[unresolved-attribute]

ARENA_MIN_X = -4.0
ARENA_MAX_X = 6.0
ARENA_HALF_WIDTH = 2.2
WALL_HALF_THICKNESS = 0.1
WALL_HEIGHT = 1.0

PIVOT = (3.0, 0.0, 0.42)
PLANK_HALF_SIZE = (0.55, 1.5, 0.06)
PLANK_MASS = 8.0
HINGE_LIMIT_DEGREES = 12.0
# Begin level so the red counterweight visibly drives the initial imbalance.
INITIAL_TILT_DEGREES = 0.0
HINGE_DAMPING = 8.0
HINGE_FRICTION_LOSS = 0.5

COUNTERWEIGHT_LOCAL_POS = (0.0, 1.15, 0.24)
COUNTERWEIGHT_HALF_SIZE = (0.38, 0.25, 0.18)
COUNTERWEIGHT_MASS = 22.0

_WALL_RGBA = (0.3, 0.35, 0.4, 1.0)
_FULCRUM_RGBA = (0.25, 0.28, 0.32, 1.0)
_PLANK_RGBA = (0.15, 0.45, 0.9, 1.0)
_COUNTERWEIGHT_RGBA = (0.75, 0.18, 0.12, 1.0)


def make_see_saw_spec_fn() -> SceneSpecFn:
    """Create a damped, range-limited see-saw with a fixed counterweight."""

    def add_see_saw(spec: MjSpec) -> None:
        _add_arena_walls(spec)
        _add_fulcrum(spec)
        _add_see_saw(spec)

    return add_see_saw


def _add_arena_walls(spec: "MjSpec") -> None:
    half_wall_height = WALL_HEIGHT * 0.5
    center_x = (ARENA_MIN_X + ARENA_MAX_X) * 0.5
    half_length = (ARENA_MAX_X - ARENA_MIN_X) * 0.5

    for side, y in (("left", ARENA_HALF_WIDTH), ("right", -ARENA_HALF_WIDTH)):
        _add_wall(
            spec,
            name=f"see_saw_{side}_wall",
            pos=(center_x, y, half_wall_height),
            size=(half_length, WALL_HALF_THICKNESS, half_wall_height),
        )
    for side, x in (("back", ARENA_MIN_X), ("front", ARENA_MAX_X)):
        _add_wall(
            spec,
            name=f"see_saw_{side}_wall",
            pos=(x, 0.0, half_wall_height),
            size=(WALL_HALF_THICKNESS, ARENA_HALF_WIDTH, half_wall_height),
        )


def _add_wall(
    spec: "MjSpec",
    *,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
) -> None:
    body = spec.worldbody.add_body(name=name)
    body.pos = pos
    body.add_geom(
        name=f"{name}_collision",
        type=MJGEOM_BOX,
        size=size,
        rgba=_WALL_RGBA,
        contype=1,
        conaffinity=1,
    )


def _add_fulcrum(spec: "MjSpec") -> None:
    x, y, pivot_z = PIVOT
    half_height = (pivot_z - PLANK_HALF_SIZE[2]) * 0.5
    body = spec.worldbody.add_body(name="see_saw_fulcrum")
    body.pos = (x, y, half_height)
    body.add_geom(
        name="see_saw_fulcrum_base",
        type=MJGEOM_BOX,
        size=(PLANK_HALF_SIZE[0] + 0.14, 0.24, half_height),
        rgba=_FULCRUM_RGBA,
        # The hinge carries the plank. Keeping this visual pedestal out of the
        # contact solver prevents it from pinning the see-saw level.
        contype=0,
        conaffinity=0,
    )


def _add_see_saw(spec: "MjSpec") -> None:
    body = spec.worldbody.add_body(name="see_saw_plank")
    body.pos = PIVOT
    body.add_joint(
        name="see_saw_hinge",
        type=MJ_JOINT_HINGE,
        axis=(1.0, 0.0, 0.0),
        range=(-HINGE_LIMIT_DEGREES, HINGE_LIMIT_DEGREES),
        ref=INITIAL_TILT_DEGREES,
        damping=HINGE_DAMPING,
        frictionloss=HINGE_FRICTION_LOSS,
    )
    body.add_geom(
        name="see_saw_plank_surface",
        type=MJGEOM_BOX,
        size=PLANK_HALF_SIZE,
        mass=PLANK_MASS,
        friction=(1.0, 0.02, 0.002),
        rgba=_PLANK_RGBA,
        contype=1,
        conaffinity=1,
    )

    counterweight = body.add_geom(
        name="see_saw_counterweight",
        type=MJGEOM_BOX,
        pos=COUNTERWEIGHT_LOCAL_POS,
        size=COUNTERWEIGHT_HALF_SIZE,
        mass=COUNTERWEIGHT_MASS,
        friction=(1.0, 0.02, 0.002),
        rgba=_COUNTERWEIGHT_RGBA,
        contype=1,
        conaffinity=1,
    )
    counterweight.quat = (1.0, 0.0, 0.0, 0.0)


def hinge_limit_radians() -> float:
    """Return the compiled joint limit in radians for tests and diagnostics."""

    return math.radians(HINGE_LIMIT_DEGREES)
