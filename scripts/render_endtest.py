#!/usr/bin/env python3
"""Render endtest and mark the centers of its green foot targets."""

import os
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from mjlab_scout.config import ScoutConfig
from mjlab_scout.runtime import ScoutRuntime

OUTPUT_DIR = Path(os.environ.get("ENDTEST_OUTPUT_DIR", "artifacts"))
RAW_PATH = OUTPUT_DIR / "endtest-overhead.jpg"
DEBUG_PATH = OUTPUT_DIR / "endtest-overhead-targets.png"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = ScoutRuntime(
        ScoutConfig(
            device=os.environ.get("DEVICE", "cuda:0"),
            image_width=1280,
            image_height=720,
            preview_seconds=0,
        )
    )
    try:
        runtime.load_task("endtest")
        image = runtime.capture_view("agent").image
    finally:
        runtime.close()

    RAW_PATH.write_bytes(image)
    frame = iio.imread(BytesIO(image), extension=".jpg")
    centers = _green_target_centers(frame)
    if len(centers) != 2:
        raise RuntimeError(f"Expected two green targets, found {len(centers)}")
    debug = _draw_target_dots(frame, centers)
    iio.imwrite(DEBUG_PATH, debug, extension=".png")
    print(RAW_PATH)
    print(DEBUG_PATH)


def _green_target_centers(image: np.ndarray) -> list[tuple[int, int]]:
    red, green, blue = (
        image[..., channel].astype(np.int16) for channel in range(3)
    )
    mask = (green > 130) & (green > red * 1.5) & (green > blue * 1.2)
    components = _connected_components(mask)
    targets = sorted(components, key=len, reverse=True)[:2]
    if len(targets) != 2 or len(targets[-1]) < 100:
        return []
    return [
        (
            round(float(np.mean([row for row, _ in component]))),
            round(float(np.mean([column for _, column in component]))),
        )
        for component in targets
    ]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    remaining = mask.copy()
    components: list[list[tuple[int, int]]] = []
    for row, column in np.argwhere(remaining):
        if not remaining[row, column]:
            continue
        component: list[tuple[int, int]] = []
        pending = [(int(row), int(column))]
        remaining[row, column] = False
        while pending:
            current_row, current_column = pending.pop()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and remaining[next_row, next_column]
                ):
                    remaining[next_row, next_column] = False
                    pending.append((next_row, next_column))
        components.append(component)
    return components


def _draw_target_dots(
    image: np.ndarray, centers: list[tuple[int, int]]
) -> np.ndarray:
    debug = image.copy()
    height, width = debug.shape[:2]
    radius = max(18, min(height, width) // 24)
    rows, columns = np.ogrid[:height, :width]
    for row, column in centers:
        distance_squared = (rows - row) ** 2 + (columns - column) ** 2
        debug[distance_squared <= radius**2] = (235, 30, 30)
        ring = (radius + 4) ** 2
        debug[(distance_squared <= ring) & (distance_squared > radius**2)] = (
            255,
            255,
            255,
        )
    return debug


if __name__ == "__main__":
    main()
