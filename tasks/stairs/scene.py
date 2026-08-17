from __future__ import annotations

import random
from typing import TYPE_CHECKING

import mujoco

from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]

ARENA_MIN_X = -2.5
ARENA_MAX_X = 5.5
ARENA_HALF_WIDTH = 2.5
WALL_HALF_THICKNESS = 0.1
WALL_HEIGHT = 1.2

PLATFORM_HEIGHT = 0.75
PLATFORM_CAP_THICKNESS = 0.025
PLATFORM_BACK_X = 3.7
PLATFORM_FRONT_X = ARENA_MAX_X - WALL_HALF_THICKNESS
PLATFORM_HALF_WIDTH = 1.4

LOWER_STEP_HALF_SIZE = (0.25, 0.45, 0.15)
UPPER_STEP_HALF_SIZE = (0.5, 0.45, 0.2)
LOWER_STEP_MASS = 4.0
UPPER_STEP_MASS = 7.0
LOWER_RAISED_TREAD_FRACTION = 0.5
UPPER_RAISED_TREAD_FRACTION = 0.75

STEP_FRICTION = (0.9, 0.02, 0.002)
STEP_SPAWN_MIN_X = 0.6
STEP_SPAWN_MAX_X = 2.9
STEP_SPAWN_MARGIN = 0.2

_WALL_RGBA = (0.3, 0.35, 0.4, 1.0)
_PLATFORM_TOP_RGBA = (0.15, 0.8, 0.3, 1.0)
_STEP_RGBA = (0.95, 0.55, 0.1, 1.0)


def make_stairs_spec_fn(*, seed: int | None = None) -> SceneSpecFn:
    """Create a walled arena with a raised platform and two movable steps."""

    lower_start, upper_start = _sample_step_starts(seed)

    def add_stairs(spec: MjSpec) -> None:
        _add_arena_walls(spec)
        _add_platform(spec)
        _add_step(
            spec,
            name="stairs_small_block",
            center=lower_start,
            half_size=LOWER_STEP_HALF_SIZE,
            mass=LOWER_STEP_MASS,
            raised_tread_fraction=LOWER_RAISED_TREAD_FRACTION,
        )
        _add_step(
            spec,
            name="stairs_large_block",
            center=upper_start,
            half_size=UPPER_STEP_HALF_SIZE,
            mass=UPPER_STEP_MASS,
            raised_tread_fraction=UPPER_RAISED_TREAD_FRACTION,
        )
        _add_small_block_grasps(spec)

    return add_stairs


def _add_arena_walls(spec: "MjSpec") -> None:
    half_wall_height = WALL_HEIGHT * 0.5
    center_x = (ARENA_MIN_X + ARENA_MAX_X) * 0.5
    half_length = (ARENA_MAX_X - ARENA_MIN_X) * 0.5

    for side, y in (("left", ARENA_HALF_WIDTH), ("right", -ARENA_HALF_WIDTH)):
        _add_wall(
            spec,
            name=f"stairs_{side}_wall",
            pos=(center_x, y, half_wall_height),
            size=(half_length, WALL_HALF_THICKNESS, half_wall_height),
        )
    for side, x in (("back", ARENA_MIN_X), ("front", ARENA_MAX_X)):
        _add_wall(
            spec,
            name=f"stairs_{side}_wall",
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
    body = spec.worldbody.add_body(name="stairs_platform")
    body.pos = (center_x, 0.0, 0.0)
    body.add_geom(
        name="stairs_platform_base",
        type=MJGEOM_BOX,
        pos=(0.0, 0.0, base_height * 0.5),
        size=(half_length, PLATFORM_HALF_WIDTH, base_height * 0.5),
        rgba=_WALL_RGBA,
        contype=1,
        conaffinity=1,
    )
    body.add_geom(
        name="stairs_platform_goal",
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
    raised_tread_fraction: float,
) -> None:
    half_x, half_y, half_z = half_size
    raised_half_x = half_x * raised_tread_fraction
    lower_mass = mass / (1.0 + raised_tread_fraction)
    x, y = center
    body = spec.worldbody.add_body(name=name)
    body.pos = (x, y, half_z)
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(
        name=f"{name}_lower_tread",
        type=MJGEOM_BOX,
        pos=(0.0, 0.0, -half_z * 0.5),
        size=(half_x, half_y, half_z * 0.5),
        mass=lower_mass,
        friction=STEP_FRICTION,
        rgba=_STEP_RGBA,
        contype=1,
        conaffinity=1,
    )
    body.add_geom(
        name=f"{name}_raised_tread",
        type=MJGEOM_BOX,
        pos=(half_x - raised_half_x, 0.0, half_z * 0.5),
        size=(raised_half_x, half_y, half_z * 0.5),
        mass=mass - lower_mass,
        friction=STEP_FRICTION,
        rgba=_STEP_RGBA,
        contype=1,
        conaffinity=1,
    )


def _add_small_block_grasps(spec: "MjSpec") -> None:
    """Add the stairs-only rest and palm welds for the small block."""

    common = {
        "type": mujoco.mjtEq.mjEQ_WELD,  # ty: ignore[unresolved-attribute]
        "objtype": mujoco.mjtObj.mjOBJ_BODY,  # ty: ignore[unresolved-attribute]
    }
    spec.add_equality(
        name="stairs_small_block_rest_weld",
        name1="stairs_small_block",
        active=True,
        **common,
    )
    for side in ("left", "right"):
        spec.add_equality(
            name=f"stairs_small_block_{side}_hand_weld",
            name1=f"robot/{side}_wrist_yaw_link",
            name2="stairs_small_block",
            active=False,
            **common,
        )


def _sample_step_starts(
    seed: int | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    rng = random.Random(seed)
    small = _sample_start(rng, LOWER_STEP_HALF_SIZE)
    for _ in range(100):
        large = _sample_start(rng, UPPER_STEP_HALF_SIZE)
        if not _footprints_overlap(
            small,
            LOWER_STEP_HALF_SIZE,
            large,
            UPPER_STEP_HALF_SIZE,
            margin=STEP_SPAWN_MARGIN,
        ):
            return small, large
    raise RuntimeError("Could not place both stair blocks without overlap")


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
        abs(first[0] - second[0]) < first_half_size[0] + second_half_size[0] + margin
        and abs(first[1] - second[1])
        < first_half_size[1] + second_half_size[1] + margin
    )
