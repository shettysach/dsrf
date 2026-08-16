from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
import numpy as np


class GridVideoRecorder:
    """Stream selected Pygame frames to a compact MP4 recording."""

    def __init__(self, path: Path, *, fps: int, hold_frames: int) -> None:
        if fps < 1 or hold_frames < 1:
            raise ValueError("fps and hold_frames must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.hold_frames = hold_frames
        self._writer = iio.get_writer(
            str(path), fps=fps, codec="libx264", macro_block_size=1
        )
        self._closed = False

    def write(self, rgb: np.ndarray) -> None:
        frame = np.asarray(rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame [H, W, 3], got {frame.shape}")
        for _ in range(self.hold_frames):
            self._writer.append_data(frame)

    def close(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True
