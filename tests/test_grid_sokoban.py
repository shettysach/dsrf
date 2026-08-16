from __future__ import annotations

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

        assert app.window.get_size() == (504, 608)
        assert image.shape == (504, 504, 3)
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
    assert available_layouts() == ("straight", "turn", "two-box")
    for name in available_layouts():
        board = GridSokoban(make_layout(name))
        assert (board.rows - 2, board.cols - 2) == (5, 5)
