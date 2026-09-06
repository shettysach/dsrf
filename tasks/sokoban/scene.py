"""Physical, tile-for-tile versions of the Sokoban evaluation boards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import mujoco

from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]


Position = tuple[int, int]
MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]
MJ_JOINT_SLIDE = mujoco.mjtJoint.mjJNT_SLIDE  # ty: ignore[unresolved-attribute]

# One logical Sokoban cell is one metre square. The cube leaves 0.30 m of
# clearance to a neighbouring wall, preserving a one-cell corridor's topology.
GRID_WIDTH = GRID_HEIGHT = 8
TILE_SIZE = 1.0
BOX_HALF_SIZE = 0.35
BOX_MASS = 0.5
BOX_SLIDE_DAMPING = 0.8
BOX_FRICTION = (0.75, 0.01, 0.001)
GOAL_HALF_SIZE = 0.46
WALL_HALF_HEIGHT = 0.6
WALL_HALF_SIZE = TILE_SIZE * 0.5

_BOX_RGBA = (0.95, 0.55, 0.1, 1.0)
_GOAL_RGBA = (0.15, 0.8, 0.3, 0.55)
_WALL_RGBA = (0.35, 0.4, 0.45, 1.0)


@dataclass(frozen=True)
class SokobanLevel:
    """An 8×8 board using the same symbols as ``evals/sokoban_eval``."""

    number: int
    name: str
    rows: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != GRID_HEIGHT or any(
            len(row) != GRID_WIDTH for row in self.rows
        ):
            raise ValueError("Sokoban levels must be 8×8")
        if any(tile not in "# .$@" for row in self.rows for tile in row):
            raise ValueError("Unsupported Sokoban tile")
        box_count = sum(row.count("$") for row in self.rows)
        if (
            box_count not in (2, 3)
            or sum(row.count(".") for row in self.rows) != box_count
        ):
            raise ValueError(
                "Each level requires one goal for every two or three boxes"
            )
        if sum(row.count("@") for row in self.rows) != 1:
            raise ValueError("Each level requires exactly one start tile")


# Local on purpose: tasks remain runnable if the adjacent evaluation application
# is not installed. These rows are copied exactly from evals/sokoban_eval/env.py.
LEVELS = (
    SokobanLevel(
        1,
        "First Pushes",
        (
            "########",
            "# .    #",
            "# $    #",
            "#   @$.#",
            "#      #",
            "#      #",
            "#      #",
            "########",
        ),
    ),
    SokobanLevel(
        2,
        "Two Directions",
        (
            "########",
            "# .    #",
            "# $    #",
            "#   $ .#",
            "#  @   #",
            "#      #",
            "#      #",
            "########",
        ),
    ),
    SokobanLevel(
        3,
        "Left Delivery",
        (
            "########",
            "# .    #",
            "# $    #",
            "#      #",
            "#.$@   #",
            "#      #",
            "#      #",
            "########",
        ),
    ),
    SokobanLevel(
        4,
        "Outside Goals",
        (
            "########",
            "#      #",
            "#. $ $.#",
            "#      #",
            "#   @  #",
            "#      #",
            "#      #",
            "########",
        ),
    ),
    SokobanLevel(
        5,
        "Top and Bottom",
        (
            "########",
            "# .#   #",
            "# $    #",
            "#      #",
            "#    $ #",
            "#   @  #",
            "#    . #",
            "########",
        ),
    ),
    SokobanLevel(
        6,
        "Wide Delivery",
        (
            "########",
            "#.     #",
            "#      #",
            "#      #",
            "# $  $.#",
            "#   @  #",
            "#      #",
            "########",
        ),
    ),
    SokobanLevel(
        7,
        "Three Lanes",
        (
            "########",
            "#.     #",
            "#     .#",
            "# $ $ $#",
            "#   @  #",
            "#      #",
            "#  .   #",
            "########",
        ),
    ),
    SokobanLevel(
        8,
        "Three Deliveries",
        (
            "########",
            "#.     #",
            "#      #",
            "#   $ .#",
            "# $  $ #",
            "#   @  #",
            "#  .   #",
            "########",
        ),
    ),
    SokobanLevel(
        9,
        "Crossing Paths",
        (
            "########",
            "#.     #",
            "#      #",
            "#      #",
            "# $$ $.#",
            "#   @  #",
            "#  .   #",
            "########",
        ),
    ),
    SokobanLevel(
        10,
        "Final Arrangement",
        (
            "########",
            "#.     #",
            "#   #  #",
            "# $ #$.#",
            "#  .$  #",
            "#   @  #",
            "#      #",
            "########",
        ),
    ),
)


def get_level(number: int) -> SokobanLevel:
    if not 1 <= number <= len(LEVELS):
        raise ValueError(f"Sokoban level must be in 1..{len(LEVELS)}")
    return LEVELS[number - 1]


def grid_to_world(cell: Position) -> tuple[float, float]:
    """Map a board column,row to the centre of its world-space tile."""
    column, row = cell
    return ((column - 3.5) * TILE_SIZE, (3.5 - row) * TILE_SIZE)


@dataclass(frozen=True)
class LevelPositions:
    walls: tuple[Position, ...]
    goals: tuple[Position, ...]
    boxes: tuple[Position, ...]
    player: Position


def level_positions(level: SokobanLevel) -> LevelPositions:
    """Extract board locations in row-major order for stable body naming."""
    walls: list[Position] = []
    goals: list[Position] = []
    boxes: list[Position] = []
    player: Position | None = None
    for row, tiles in enumerate(level.rows):
        for column, tile in enumerate(tiles):
            cell = (column, row)
            if tile == "#":
                walls.append(cell)
            elif tile == ".":
                goals.append(cell)
            elif tile == "$":
                boxes.append(cell)
            elif tile == "@":
                player = cell
    return LevelPositions(
        tuple(walls), tuple(goals), tuple(boxes), cast(Position, player)
    )


def make_sokoban_spec_fn(level: int = 1) -> SceneSpecFn:
    """Build the physical scene for one of the ten fixed evaluation boards."""
    positions = level_positions(get_level(level))

    def add_sokoban(spec: MjSpec) -> None:
        for index, cell in enumerate(positions.walls, 1):
            _add_wall(spec, index=index, center=grid_to_world(cell))
        for index, cell in enumerate(positions.goals, 1):
            _add_goal(spec, index=index, center=grid_to_world(cell))
        for index, cell in enumerate(positions.boxes, 1):
            _add_box(spec, index=index, center=grid_to_world(cell))

    return add_sokoban


def _add_wall(spec: "MjSpec", *, index: int, center: tuple[float, float]) -> None:
    x, y = center
    name = f"sokoban_wall_{index}"
    body = spec.worldbody.add_body(name=name)
    body.pos = (x, y, WALL_HALF_HEIGHT)
    body.add_geom(
        name=f"{name}_collision",
        type=MJGEOM_BOX,
        size=(WALL_HALF_SIZE, WALL_HALF_SIZE, WALL_HALF_HEIGHT),
        rgba=_WALL_RGBA,
        contype=1,
        conaffinity=1,
    )


def _add_goal(spec: "MjSpec", *, index: int, center: tuple[float, float]) -> None:
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


def _add_box(spec: "MjSpec", *, index: int, center: tuple[float, float]) -> None:
    name = f"sokoban_box_{index}"
    x, y = center
    body = spec.worldbody.add_body(name=name)
    body.pos = (x, y, BOX_HALF_SIZE)
    body.add_joint(
        name=f"{name}_x",
        type=MJ_JOINT_SLIDE,
        axis=(1.0, 0.0, 0.0),
        damping=BOX_SLIDE_DAMPING,
    )
    body.add_joint(
        name=f"{name}_y",
        type=MJ_JOINT_SLIDE,
        axis=(0.0, 1.0, 0.0),
        damping=BOX_SLIDE_DAMPING,
    )
    body.add_geom(
        name=f"{name}_collision",
        type=MJGEOM_BOX,
        size=(BOX_HALF_SIZE, BOX_HALF_SIZE, BOX_HALF_SIZE),
        mass=BOX_MASS,
        friction=BOX_FRICTION,
        rgba=_BOX_RGBA,
        contype=1,
        conaffinity=1,
    )
