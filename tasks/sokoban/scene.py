from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]
MJ_JOINT_SLIDE = mujoco.mjtJoint.mjJNT_SLIDE  # ty: ignore[unresolved-attribute]

BOX_HALF_SIZE = 0.3
BOX_MASS = 0.5
GOAL_HALF_SIZE = 0.48
ARENA_MIN_X = -1.5
ARENA_MAX_X = 5.5
ARENA_HALF_WIDTH = 2.5

BOX_STARTS = ((1.25, -0.8), (1.25, 0.8))
GOAL_CENTERS = ((4.25, -0.8), (4.25, 0.8))

_BOX_RGBA = (0.95, 0.55, 0.1, 1.0)
_GOAL_RGBA = (0.15, 0.8, 0.3, 0.55)
_WALL_RGBA = (0.35, 0.4, 0.45, 1.0)


def make_sokoban_spec_fn() -> SceneSpecFn:
    """Create a compact two-box Sokoban arena without scouting cameras."""

    def add_sokoban(spec: MjSpec) -> None:
        _add_arena_walls(spec)
        for index, center in enumerate(GOAL_CENTERS, 1):
            _add_goal(spec, index=index, center=center)
        for index, center in enumerate(BOX_STARTS, 1):
            _add_box(spec, index=index, center=center)

    return add_sokoban


def _add_arena_walls(spec: "MjSpec") -> None:
    wall_height = 0.6
    wall_thickness = 0.1
    center_x = (ARENA_MIN_X + ARENA_MAX_X) * 0.5
    half_length = (ARENA_MAX_X - ARENA_MIN_X) * 0.5

    for side, y in (("left", ARENA_HALF_WIDTH), ("right", -ARENA_HALF_WIDTH)):
        _add_wall(
            spec,
            name=f"sokoban_{side}_wall",
            pos=(center_x, y, wall_height),
            size=(half_length, wall_thickness, wall_height),
        )
    for side, x in (("back", ARENA_MIN_X), ("front", ARENA_MAX_X)):
        _add_wall(
            spec,
            name=f"sokoban_{side}_wall",
            pos=(x, 0.0, wall_height),
            size=(wall_thickness, ARENA_HALF_WIDTH, wall_height),
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


def _add_goal(
    spec: "MjSpec",
    *,
    index: int,
    center: tuple[float, float],
) -> None:
    x, y = center
    spec.worldbody.add_geom(
        name=f"sokoban_goal_{index}",
        type=MJGEOM_BOX,
        pos=(x, y, 0.006),
        size=(GOAL_HALF_SIZE, GOAL_HALF_SIZE, 0.005),
        rgba=_GOAL_RGBA,
        contype=0,
        conaffinity=0,
        mass=0.0,
    )


def _add_box(
    spec: "MjSpec",
    *,
    index: int,
    center: tuple[float, float],
) -> None:
    name = f"sokoban_box_{index}"
    x, y = center
    body = spec.worldbody.add_body(name=name)
    body.pos = (x, y, BOX_HALF_SIZE)
    body.add_joint(
        name=f"{name}_x",
        type=MJ_JOINT_SLIDE,
        axis=(1.0, 0.0, 0.0),
        damping=0.1,
    )
    body.add_joint(
        name=f"{name}_y",
        type=MJ_JOINT_SLIDE,
        axis=(0.0, 1.0, 0.0),
        damping=0.1,
    )
    body.add_geom(
        name=f"{name}_collision",
        type=MJGEOM_BOX,
        size=(BOX_HALF_SIZE, BOX_HALF_SIZE, BOX_HALF_SIZE),
        mass=BOX_MASS,
        friction=(0.25, 0.005, 0.0001),
        rgba=_BOX_RGBA,
        contype=1,
        conaffinity=1,
    )
