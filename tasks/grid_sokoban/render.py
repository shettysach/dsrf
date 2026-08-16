from __future__ import annotations

from io import BytesIO

import imageio.v3 as iio
import numpy as np

from tasks.grid_sokoban.env import GridSokoban, Position


def render_jpeg(
    board: GridSokoban, *, cell_pixels: int = 96, quality: int = 90
) -> bytes:
    """Render a fixed top-down board image without textual state annotations."""
    if cell_pixels < 24:
        raise ValueError("cell_pixels must be at least 24")
    image = np.full(
        (board.rows * cell_pixels, board.cols * cell_pixels, 3),
        (242, 239, 228),
        dtype=np.uint8,
    )
    for row in range(board.rows):
        for col in range(board.cols):
            top, left = row * cell_pixels, col * cell_pixels
            tile = image[top : top + cell_pixels, left : left + cell_pixels]
            if (row, col) in board.walls:
                tile[:] = (65, 72, 80)
            else:
                tile[0:2, :] = (210, 206, 194)
                tile[:, 0:2] = (210, 206, 194)
    for goal in board.goals:
        _fill_cell(image, goal, cell_pixels, (71, 166, 94))
    for box in board.boxes:
        _box(image, box, cell_pixels, (224, 156, 43))
    _circle(image, board.player, cell_pixels, (52, 119, 204), radius=0.27)
    buffer = BytesIO()
    iio.imwrite(buffer, image, extension=".jpg", quality=quality)
    return buffer.getvalue()


def _circle(
    image: np.ndarray,
    position: Position,
    cell_pixels: int,
    color: tuple[int, int, int],
    *,
    radius: float,
) -> None:
    row, col = position
    y, x = np.ogrid[:cell_pixels, :cell_pixels]
    center = (cell_pixels - 1) / 2
    mask = (x - center) ** 2 + (y - center) ** 2 <= (cell_pixels * radius) ** 2
    tile = image[
        row * cell_pixels : (row + 1) * cell_pixels,
        col * cell_pixels : (col + 1) * cell_pixels,
    ]
    tile[mask] = color


def _box(
    image: np.ndarray,
    position: Position,
    cell_pixels: int,
    color: tuple[int, int, int],
    *,
    inset_ratio: float = 0.19,
) -> None:
    row, col = position
    inset = round(cell_pixels * inset_ratio)
    top, left = row * cell_pixels + inset, col * cell_pixels + inset
    image[
        top : top + cell_pixels - 2 * inset, left : left + cell_pixels - 2 * inset
    ] = color


def _fill_cell(
    image: np.ndarray,
    position: Position,
    cell_pixels: int,
    color: tuple[int, int, int],
) -> None:
    row, col = position
    image[
        row * cell_pixels : (row + 1) * cell_pixels,
        col * cell_pixels : (col + 1) * cell_pixels,
    ] = color
