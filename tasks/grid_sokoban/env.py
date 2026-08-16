from __future__ import annotations

from dataclasses import dataclass

Position = tuple[int, int]
Action = str

_ACTIONS = frozenset({"up", "down", "left", "right", "reset"})
_OFFSETS: dict[str, Position] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# The boards deliberately avoid irreversible corner pushes. They progress from
# direct pushes to repositioning, then reproduce the earlier physical task's
# open-floor box with a goal at an arena edge.
_LAYOUTS: dict[str, tuple[str, ...]] = {
    "straight": (
        "#######",
        "#     #",
        "#     #",
        "# @$. #",
        "#     #",
        "#     #",
        "#######",
    ),
    "turn": (
        "#######",
        "#     #",
        "#  $. #",
        "#  @  #",
        "#     #",
        "#     #",
        "#######",
    ),
    "two-box": (
        "#######",
        "# . . #",
        "# $ $ #",
        "#  @  #",
        "#     #",
        "#     #",
        "#######",
    ),
    # Analogous to the prior physical Sokoban task: a free box must be pushed
    # across open floor onto a goal at the right arena edge.
    "edge-right": (
        "#######",
        "#     #",
        "#     #",
        "#@ $ .#",
        "#     #",
        "#     #",
        "#######",
    ),
    "edge-top": (
        "#######",
        "#  .  #",
        "#     #",
        "#  $  #",
        "#  @  #",
        "#     #",
        "#######",
    ),
    "two-edge": (
        "#######",
        "#.   .#",
        "#     #",
        "#$   $#",
        "#@    #",
        "#     #",
        "#######",
    ),
}


@dataclass(frozen=True)
class StepResult:
    action: Action
    moved: bool
    pushed: bool
    reset: bool
    solved: bool


class GridSokoban:
    """A fully deterministic Sokoban state machine with no physics dependency."""

    def __init__(self, layout: tuple[str, ...]) -> None:
        if not layout or len({len(row) for row in layout}) != 1:
            raise ValueError("Layout must be a non-empty rectangle")
        self.rows = len(layout)
        self.cols = len(layout[0])
        self.walls: set[Position] = set()
        self.goals: set[Position] = set()
        boxes: set[Position] = set()
        player: Position | None = None
        valid = {"#", " ", ".", "$", "@", "*", "+"}
        for row, line in enumerate(layout):
            for col, tile in enumerate(line):
                if tile not in valid:
                    raise ValueError(f"Unsupported layout tile {tile!r}")
                position = (row, col)
                if tile == "#":
                    self.walls.add(position)
                if tile in {".", "*", "+"}:
                    self.goals.add(position)
                if tile in {"$", "*"}:
                    boxes.add(position)
                if tile in {"@", "+"}:
                    if player is not None:
                        raise ValueError("Layout must contain exactly one player")
                    player = position
        if player is None:
            raise ValueError("Layout must contain exactly one player")
        if not boxes or len(boxes) != len(self.goals):
            raise ValueError("Layout must contain equal non-zero box and goal counts")
        if boxes & self.walls or player in self.walls:
            raise ValueError("Walls cannot contain a player or box")
        self._initial_player = player
        self._initial_boxes = frozenset(boxes)
        self.player = player
        self.boxes = boxes

    @property
    def solved(self) -> bool:
        return self.boxes == self.goals

    def reset(self) -> StepResult:
        self.player = self._initial_player
        self.boxes = set(self._initial_boxes)
        return StepResult(
            "reset", moved=False, pushed=False, reset=True, solved=self.solved
        )

    def step(self, action: Action) -> StepResult:
        if action not in _ACTIONS:
            raise ValueError(
                f"Unsupported action {action!r}; expected one of {_ACTIONS}"
            )
        if action == "reset":
            return self.reset()
        row_delta, col_delta = _OFFSETS[action]
        next_position = (self.player[0] + row_delta, self.player[1] + col_delta)
        if next_position in self.walls:
            return StepResult(
                action, moved=False, pushed=False, reset=False, solved=self.solved
            )
        if next_position not in self.boxes:
            self.player = next_position
            return StepResult(
                action, moved=True, pushed=False, reset=False, solved=self.solved
            )
        box_destination = (
            next_position[0] + row_delta,
            next_position[1] + col_delta,
        )
        if box_destination in self.walls or box_destination in self.boxes:
            return StepResult(
                action, moved=False, pushed=False, reset=False, solved=self.solved
            )
        self.boxes.remove(next_position)
        self.boxes.add(box_destination)
        self.player = next_position
        return StepResult(
            action, moved=True, pushed=True, reset=False, solved=self.solved
        )


def available_layouts() -> tuple[str, ...]:
    return tuple(_LAYOUTS)


def make_layout(name: str) -> tuple[str, ...]:
    try:
        return _LAYOUTS[name]
    except KeyError:
        raise ValueError(
            f"Unknown grid Sokoban layout {name!r}. Available: {', '.join(_LAYOUTS)}"
        ) from None
