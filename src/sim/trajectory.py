from __future__ import annotations

from io import BytesIO

import imageio.v3 as iio
import numpy as np


class TrajectoryRenderer:
    """Render the robot's observed 2D path without exposing world coordinates."""

    def __init__(self, *, resolution: int = 256, max_points: int = 500) -> None:
        if resolution < 16:
            raise ValueError("Trajectory resolution must be at least 16")
        if max_points < 1:
            raise ValueError("Trajectory max_points must be positive")
        self.resolution = resolution
        self.max_points = max_points
        self._points: list[tuple[float, float]] = []

    def reset(self) -> None:
        self._points.clear()

    def append(self, position_xy: tuple[float, float]) -> None:
        x, y = position_xy
        if not np.isfinite((x, y)).all():
            raise ValueError("Trajectory point must be finite")
        self._points.append((float(x), float(y)))
        if len(self._points) > self.max_points:
            self._points = [self._points[0], *self._points[-(self.max_points - 1) :]]

    def render_png(self) -> bytes:
        image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        if self._points:
            pixels = _fit_to_pixels(np.asarray(self._points, dtype=np.float32), self.resolution)
            for start, end in zip(pixels, pixels[1:], strict=False):
                _draw_line(image, start, end)
            _draw_dot(image, pixels[-1])
        output = BytesIO()
        iio.imwrite(output, image, extension=".png")
        return output.getvalue()


def _fit_to_pixels(points: np.ndarray, resolution: int) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = max(float((maximum - minimum).max()), 1.0)
    center = (minimum + maximum) / 2.0
    margin = 8
    scale = (resolution - 2 * margin) / span
    pixels = (points - center) * scale + (resolution - 1) / 2.0
    pixels[:, 1] = resolution - 1 - pixels[:, 1]
    return np.rint(pixels).astype(np.int32)


def _draw_line(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    distance = max(abs(int(end[0] - start[0])), abs(int(end[1] - start[1])))
    for fraction in np.linspace(0.0, 1.0, distance + 1):
        point = np.rint(start + (end - start) * fraction).astype(np.int32)
        _draw_dot(image, point)


def _draw_dot(image: np.ndarray, point: np.ndarray) -> None:
    x, y = int(point[0]), int(point[1])
    height, width = image.shape[:2]
    for row in range(max(0, y - 1), min(height, y + 2)):
        for column in range(max(0, x - 1), min(width, x + 2)):
            image[row, column] = 255
