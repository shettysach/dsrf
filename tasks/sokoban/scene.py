"""Physical, tile-for-tile versions of the Sokoban evaluation boards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

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
# Continuous pushes do not stop perfectly at a cell centre.  The visible green
# square is the interaction region, so a box is complete when its centre is in
# that square; this is stable under small physical push error and matches the
# visual rule the VLM can apply.
WALL_HALF_HEIGHT = 0.6
OUTER_WALL_HALF_THICKNESS = 0.1

_BOX_RGBA = (0.95, 0.55, 0.1, 1.0)
COMPLETED_BOX_RGBA = (0.05, 0.35, 0.12, 1.0)
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


class SokobanCompletionVisualizer:
    """Colors a box dark green when its centre is inside a goal tile."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._box_ids = _named_geom_ids(model, "sokoban_box_", "_collision")
        goal_ids = _named_geom_ids(model, "sokoban_goal_", "")
        self._goal_centers = np.asarray(model.geom_pos[goal_ids, :2], dtype=float)

    def update(self, data: Any) -> bool:
        geom_positions = _as_numpy(data.geom_xpos)
        if geom_positions.ndim == 3:
            geom_positions = geom_positions[0]
        box_centers = geom_positions[list(self._box_ids), :2]
        # The goal square itself is the visual completion region.  Requiring
        # whole-footprint overlap made diagonal but visibly valid placements
        # remain yellow.
        delta = np.abs(box_centers[:, None, :] - self._goal_centers[None, :, :])
        within_goal = np.all(delta <= GOAL_HALF_SIZE, axis=2)
        box_on_goal = np.any(within_goal, axis=1)
        for geom_id, is_completed in zip(self._box_ids, box_on_goal, strict=True):
            self._model.geom_rgba[geom_id] = (
                COMPLETED_BOX_RGBA if is_completed else _BOX_RGBA
            )
        # Equal box/goal counts are enforced by ``SokobanLevel``. Requiring
        # every goal to be occupied prevents two boxes sharing one goal from
        # being reported as a solved board.
        return bool(np.any(within_goal, axis=0).all())


def _named_geom_ids(model: Any, prefix: str, suffix: str) -> tuple[int, ...]:
    ids: list[int] = []
    index = 1
    while True:
        name = f"{prefix}{index}{suffix}"
        try:
            ids.append(model.geom(name).id)
        except KeyError:
            return tuple(ids)
        index += 1


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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
    assert player is not None  # guaranteed by SokobanLevel validation
    return LevelPositions(tuple(walls), tuple(goals), tuple(boxes), player)


def make_sokoban_spec_fn(level: int = 1) -> SceneSpecFn:
    """Build the physical scene for one of the ten fixed evaluation boards."""
    positions = level_positions(get_level(level))

    def add_sokoban(spec: MjSpec) -> None:
        _add_outer_walls(spec)
        _add_interior_wall_runs(spec, positions.walls)
        for index, cell in enumerate(positions.goals, 1):
            _add_goal(spec, index=index, center=grid_to_world(cell))
        for index, cell in enumerate(positions.boxes, 1):
            _add_box(spec, index=index, center=grid_to_world(cell))

    return add_sokoban


def _is_outer_cell(cell: Position) -> bool:
    column, row = cell
    return column in {0, GRID_WIDTH - 1} or row in {0, GRID_HEIGHT - 1}


def _add_outer_walls(spec: "MjSpec") -> None:
    # The discrete outer cells are walls. Put the thin physical boundary at
    # their inner edge, flush with the playable cells, rather than at the
    # board's outer edge where it would leave a one-cell visual gap.  Each side
    # ends at its corner instead of extending through the perpendicular side:
    # overlapping long slabs create conspicuous plus-shaped joins in the
    # overhead render.
    playable_half_extent = GRID_WIDTH * TILE_SIZE * 0.5 - TILE_SIZE
    boundary_center = playable_half_extent + OUTER_WALL_HALF_THICKNESS
    for name, pos, size in (
        (
            "sokoban_outer_north_wall",
            (0.0, boundary_center, WALL_HALF_HEIGHT),
            (playable_half_extent, OUTER_WALL_HALF_THICKNESS, WALL_HALF_HEIGHT),
        ),
        (
            "sokoban_outer_south_wall",
            (0.0, -boundary_center, WALL_HALF_HEIGHT),
            (playable_half_extent, OUTER_WALL_HALF_THICKNESS, WALL_HALF_HEIGHT),
        ),
        (
            "sokoban_outer_east_wall",
            (boundary_center, 0.0, WALL_HALF_HEIGHT),
            (OUTER_WALL_HALF_THICKNESS, playable_half_extent, WALL_HALF_HEIGHT),
        ),
        (
            "sokoban_outer_west_wall",
            (-boundary_center, 0.0, WALL_HALF_HEIGHT),
            (OUTER_WALL_HALF_THICKNESS, playable_half_extent, WALL_HALF_HEIGHT),
        ),
    ):
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


def _add_interior_wall_runs(spec: "MjSpec", walls: tuple[Position, ...]) -> None:
    """Render contiguous interior wall cells as one rectangular slab.

    Individual cell cubes leave dark seams at their shared edges.  In a tilted
    overview those seams read as plus signs, rather than as a continuous wall.
    The collision footprint remains exactly the union of the original cells.
    """
    remaining = {cell for cell in walls if not _is_outer_cell(cell)}
    index = 1
    while remaining:
        start = min(remaining, key=lambda cell: (cell[1], cell[0]))
        horizontal = _wall_run(start, (1, 0), remaining)
        vertical = _wall_run(start, (0, 1), remaining)
        run = horizontal if len(horizontal) >= len(vertical) else vertical
        for cell in run:
            remaining.remove(cell)
        _add_wall_run(spec, index=index, cells=run)
        index += 1


def _wall_run(
    start: Position, direction: Position, available: set[Position]
) -> tuple[Position, ...]:
    cells = [start]
    x, y = start
    dx, dy = direction
    while (x + dx, y + dy) in available:
        x += dx
        y += dy
        cells.append((x, y))
    return tuple(cells)


def _add_wall_run(spec: "MjSpec", *, index: int, cells: tuple[Position, ...]) -> None:
    first_x, first_y = grid_to_world(cells[0])
    last_x, last_y = grid_to_world(cells[-1])
    x, y = ((first_x + last_x) * 0.5, (first_y + last_y) * 0.5)
    half_x = (abs(last_x - first_x) + TILE_SIZE) * 0.5
    half_y = (abs(last_y - first_y) + TILE_SIZE) * 0.5
    name = f"sokoban_wall_{index}"
    body = spec.worldbody.add_body(name=name)
    body.pos = (x, y, WALL_HALF_HEIGHT)
    body.add_geom(
        name=f"{name}_collision",
        type=MJGEOM_BOX,
        size=(half_x, half_y, WALL_HALF_HEIGHT),
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
