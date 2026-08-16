from __future__ import annotations

import time
from collections import deque
from io import BytesIO

import imageio.v3 as iio
import pytest
from tasks.grid_sokoban import (
    GridSokoban,
    available_layouts,
    make_layout,
    two_box_variations,
)
from tasks.grid_sokoban.protocol import parse_action
from tasks.grid_sokoban.render import render_jpeg


def test_straight_layout_solves_with_one_push() -> None:
    board = GridSokoban(make_layout("straight"))

    result = board.step("right")

    assert result.moved and result.pushed and result.solved
    assert board.solved


def test_edge_right_recreates_the_open_floor_edge_goal_push() -> None:
    board = GridSokoban(make_layout("edge-right"))

    for action in ("right", "right", "right"):
        board.step(action)

    assert board.solved
    assert board.goals == {(3, 5)}
    assert board.boxes == {(3, 5)}


def test_box_cannot_be_pulled_or_pushed_into_a_wall() -> None:
    board = GridSokoban(
        (
            "#####",
            "#@$.#",
            "#####",
        )
    )

    assert board.step("left").moved is False
    assert board.step("right").solved
    blocked = board.step("right")

    assert not blocked.moved
    assert not blocked.pushed
    assert board.solved


def test_box_in_a_non_goal_corner_is_a_static_deadlock() -> None:
    board = GridSokoban(
        (
            "#####",
            "#$ @#",
            "# . #",
            "#####",
        )
    )

    assert board.has_non_goal_corner_box()


def test_reset_restores_the_initial_state() -> None:
    board = GridSokoban(make_layout("straight"))
    board.step("right")

    result = board.step("reset")

    assert result.reset
    assert not board.solved
    assert board.player == (3, 2)
    assert board.boxes == {(3, 3)}


def test_renderer_produces_a_top_down_jpeg() -> None:
    board = GridSokoban(make_layout("two-box"))

    image = iio.imread(BytesIO(render_jpeg(board, cell_pixels=32)), extension=".jpg")

    assert image.shape == (board.rows * 32, board.cols * 32, 3)


def test_pygame_front_end_draws_the_playable_board(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from tasks.grid_sokoban.app import SokobanApp, _surface_jpeg

    app = SokobanApp(layout_name="straight", vlm=None, auto_play=False)
    try:
        app._draw()
        image = iio.imread(BytesIO(_surface_jpeg(app.board_surface)), extension=".jpg")

        assert app.window.get_size() == (924, 608)
        assert image.shape == (504, 504, 3)
        assert app.board_surface.get_at((4 * 72 + 1, 3 * 72 + 1))[:3] == (75, 176, 104)
    finally:
        import pygame

        pygame.quit()


def test_invalid_vlm_json_is_retried_without_moving_the_board(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from tasks.grid_sokoban.app import SokobanApp

    class FakeVlm:
        def __init__(self) -> None:
            self.responses = iter(("not json", '{"action":"right"}'))
            self.feedback: list[str | None] = []

        def complete(
            self, _observation: object, *, retry_feedback: str | None = None
        ) -> str:
            self.feedback.append(retry_feedback)
            return next(self.responses)

        def commit(self, _observation: object, _command: str) -> None:
            pass

        def reset(self) -> None:
            pass

    app = SokobanApp(layout_name="straight", vlm=FakeVlm(), auto_play=True)
    try:
        app._start_vlm_request()
        for _ in range(100):
            app._consume_vlm_result()
            if app.board.solved:
                break
            time.sleep(0.001)

        assert app.board.solved
        assert app.moves == 1
        assert app.vlm.feedback[0] is None
        assert app.vlm.feedback[1] is not None
        assert "invalid" in app.vlm.feedback[1].lower()
    finally:
        import pygame

        pygame.quit()


def test_grid_action_parser_requires_one_allowed_action() -> None:
    assert parse_action('{"action":"left"}') == "left"
    assert parse_action('```json\n{"action":"right"}\n```') == "right"
    assert parse_action('```\n{"action":"down"}\n```') == "down"
    with pytest.raises(ValueError, match="exactly"):
        parse_action('{"action":"left","note":"extra"}')
    with pytest.raises(ValueError, match="Unsupported"):
        parse_action('{"action":"jump"}')


def test_grid_layouts_cover_the_curriculum() -> None:
    assert available_layouts() == (
        "straight",
        "turn",
        "two-box",
        "edge-right",
        "edge-top",
        "two-edge",
        "two-topology-01",
        "two-topology-02",
        "two-topology-03",
        "two-topology-04",
        "two-topology-05",
        "two-topology-06",
        "two-topology-07",
        "two-topology-08",
    )
    for name in available_layouts():
        board = GridSokoban(make_layout(name))
        assert (board.rows - 2, board.cols - 2) == (5, 5)


def test_two_box_variations_provide_at_least_ten_distinct_runs() -> None:
    variations = two_box_variations()

    assert [name for name, _ in variations] == [
        "two-box",
        "two-edge",
        "two-topology-01",
        "two-topology-02",
        "two-topology-03",
        "two-topology-04",
        "two-topology-05",
        "two-topology-06",
        "two-topology-07",
        "two-topology-08",
    ]
    assert len({layout for _, layout in variations}) == len(variations)
    for _, layout in variations:
        board = GridSokoban(layout)
        assert len(board.boxes) == len(board.goals) == 2
        assert _shortest_solution_length(board) is not None


def test_vlm_reset_advances_to_the_next_recorded_variation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from tasks.grid_sokoban.app import SokobanApp

    schedule = two_box_variations()[:2]
    app = SokobanApp(
        layout_name=schedule[0][0],
        vlm=None,
        auto_play=False,
        run_schedule=schedule,
    )
    try:
        app._move("reset", source="VLM")

        assert app._advance_after_frame
        assert "RESET REQUESTED" in app.status
        app._advance_run()
        assert app.run_index == 1
        assert app.layout_name == schedule[1][0]
        assert app.moves == 0
    finally:
        import pygame

        pygame.quit()


def test_manual_skip_advances_to_the_next_recorded_variation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from tasks.grid_sokoban.app import SokobanApp

    schedule = two_box_variations()[:2]
    app = SokobanApp(
        layout_name=schedule[0][0],
        vlm=None,
        auto_play=False,
        run_schedule=schedule,
    )
    try:
        app._skip_run()

        assert app._advance_after_frame
        assert "SKIPPED" in app.status
        app._advance_run()
        assert app.run_index == 1
        assert app.layout_name == schedule[1][0]
    finally:
        import pygame

        pygame.quit()


def _shortest_solution_length(board: GridSokoban) -> int | None:
    """Small test-only BFS that rejects accidentally unsolvable fixed maps."""
    start = (frozenset(board.boxes), board.player)
    frontier = deque([(start, 0)])
    visited = {start}
    while frontier:
        (boxes, player), length = frontier.popleft()
        if boxes == board.goals:
            return length
        for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_player = (player[0] + row_delta, player[1] + col_delta)
            if next_player in board.walls:
                continue
            if next_player in boxes:
                box_destination = (
                    next_player[0] + row_delta,
                    next_player[1] + col_delta,
                )
                if box_destination in board.walls or box_destination in boxes:
                    continue
                next_boxes = set(boxes)
                next_boxes.remove(next_player)
                next_boxes.add(box_destination)
                state = (frozenset(next_boxes), next_player)
            else:
                state = (boxes, next_player)
            if state not in visited:
                visited.add(state)
                frontier.append((state, length + 1))
    return None
