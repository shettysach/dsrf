from __future__ import annotations

import random
from typing import TYPE_CHECKING

import mujoco

from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]
MJ_JOINT_FREE = mujoco.mjtJoint.mjJNT_FREE  # ty: ignore[unresolved-attribute]

ARENA_MIN_X = -4.0
ARENA_MAX_X = 6.0
ARENA_HALF_WIDTH = 2.5
WALL_HALF_THICKNESS = 0.1
WALL_HEIGHT = 1.2

PLATFORM_HEIGHT = 0.75
PLATFORM_CAP_THICKNESS = 0.025
PLATFORM_BACK_X = 4.0
PLATFORM_FRONT_X = ARENA_MAX_X - WALL_HALF_THICKNESS
PLATFORM_HALF_WIDTH = 1.4

BOTTOM_STEP_HALF_SIZE = (0.55, 0.7, 0.125)
TOP_STEP_HALF_SIZE = (0.4, 0.45, 0.125)
BOTTOM_STEP_MASS = 4.0
TOP_STEP_MASS = 2.5
STEP_FRICTION = (0.9, 0.02, 0.002)
STEP_SPAWN_MIN_X = 1.0
STEP_SPAWN_MAX_X = 3.0
STEP_SPAWN_MARGIN = 0.2

_WALL_RGBA = (0.3, 0.35, 0.4, 1.0)
_PLATFORM_RGBA = _WALL_RGBA
_PLATFORM_TOP_RGBA = (0.15, 0.8, 0.3, 1.0)
_STEP_RGBA = (1.0, 0.75, 0.05, 1.0)


def make_stack_steps_spec_fn(*, seed: int | None = None) -> SceneSpecFn:
    """Create a walled arena with a goal platform and two movable step blocks."""

    bottom_start, top_start = _sample_step_starts(seed)

    def add_stack_steps(spec: MjSpec) -> None:
        _add_arena_walls(spec)
        _add_platform(spec)
        _add_step(
            spec,
            name="stack_steps_bottom_step",
            center=bottom_start,
            half_size=BOTTOM_STEP_HALF_SIZE,
            mass=BOTTOM_STEP_MASS,
        )
        _add_step(
            spec,
            name="stack_steps_top_step",
            center=top_start,
            half_size=TOP_STEP_HALF_SIZE,
            mass=TOP_STEP_MASS,
        )

    return add_stack_steps


def _add_arena_walls(spec: "MjSpec") -> None:
    half_wall_height = WALL_HEIGHT * 0.5
    center_x = (ARENA_MIN_X + ARENA_MAX_X) * 0.5
    half_length = (ARENA_MAX_X - ARENA_MIN_X) * 0.5

    for side, y in (("left", ARENA_HALF_WIDTH), ("right", -ARENA_HALF_WIDTH)):
        _add_wall(
            spec,
            name=f"stack_steps_{side}_wall",
            pos=(center_x, y, half_wall_height),
            size=(half_length, WALL_HALF_THICKNESS, half_wall_height),
        )
    for side, x in (("back", ARENA_MIN_X), ("front", ARENA_MAX_X)):
        _add_wall(
            spec,
            name=f"stack_steps_{side}_wall",
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


def _add_platform(spec: "MjSpec") -> None:
    half_length = (PLATFORM_FRONT_X - PLATFORM_BACK_X) * 0.5
    center_x = (PLATFORM_BACK_X + PLATFORM_FRONT_X) * 0.5
    base_height = PLATFORM_HEIGHT - PLATFORM_CAP_THICKNESS
    body = spec.worldbody.add_body(name="stack_steps_platform")
    body.pos = (center_x, 0.0, 0.0)
    body.add_geom(
        name="stack_steps_platform_base",
        type=MJGEOM_BOX,
        pos=(0.0, 0.0, base_height * 0.5),
        size=(half_length, PLATFORM_HALF_WIDTH, base_height * 0.5),
        rgba=_PLATFORM_RGBA,
        contype=1,
        conaffinity=1,
    )
    body.add_geom(
        name="stack_steps_platform_goal",
        type=MJGEOM_BOX,
        pos=(0.0, 0.0, PLATFORM_HEIGHT - PLATFORM_CAP_THICKNESS * 0.5),
        size=(half_length, PLATFORM_HALF_WIDTH, PLATFORM_CAP_THICKNESS * 0.5),
        rgba=_PLATFORM_TOP_RGBA,
        contype=1,
        conaffinity=1,
    )


def _add_step(
    spec: "MjSpec",
    *,
    name: str,
    center: tuple[float, float],
    half_size: tuple[float, float, float],
    mass: float,
) -> None:
    _, _, half_z = half_size
    x, y = center
    body = spec.worldbody.add_body(name=name)
    body.pos = (x, y, half_z)
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(
        name=f"{name}_pushable",
        type=MJGEOM_BOX,
        size=half_size,
        mass=mass,
        friction=STEP_FRICTION,
        rgba=_STEP_RGBA,
        contype=1,
        conaffinity=1,
    )


def _sample_step_starts(
    seed: int | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    rng = random.Random(seed)
    bottom = _sample_start(rng, BOTTOM_STEP_HALF_SIZE)
    for _ in range(100):
        top = _sample_start(rng, TOP_STEP_HALF_SIZE)
        if not _footprints_overlap(
            bottom,
            BOTTOM_STEP_HALF_SIZE,
            top,
            TOP_STEP_HALF_SIZE,
            margin=STEP_SPAWN_MARGIN,
        ):
            return bottom, top
    raise RuntimeError("Could not place both stack-steps blocks without overlap")


def _sample_start(
    rng: random.Random,
    half_size: tuple[float, float, float],
) -> tuple[float, float]:
    half_x, half_y, _ = half_size
    maximum_x = min(STEP_SPAWN_MAX_X, PLATFORM_BACK_X - half_x - STEP_SPAWN_MARGIN)
    maximum_y = ARENA_HALF_WIDTH - WALL_HALF_THICKNESS - half_y - STEP_SPAWN_MARGIN
    return (
        rng.uniform(STEP_SPAWN_MIN_X + half_x, maximum_x),
        rng.uniform(-maximum_y, maximum_y),
    )


def _footprints_overlap(
    first: tuple[float, float],
    first_half_size: tuple[float, float, float],
    second: tuple[float, float],
    second_half_size: tuple[float, float, float],
    *,
    margin: float,
) -> bool:
    return (
        abs(first[0] - second[0])
        < first_half_size[0] + second_half_size[0] + margin
        and abs(first[1] - second[1])
        < first_half_size[1] + second_half_size[1] + margin
    )
