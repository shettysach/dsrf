from __future__ import annotations

import time
from io import BytesIO

import imageio.v3 as iio
import pytest
from tasks.grid_sokoban import GridSokoban, available_layouts, make_layout
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
    )
    for name in available_layouts():
        board = GridSokoban(make_layout(name))
        assert (board.rows - 2, board.cols - 2) == (5, 5)
