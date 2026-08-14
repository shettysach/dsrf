from __future__ import annotations

from io import BytesIO

import imageio.v3 as iio
import numpy as np

from shared.messages import ProjectionContext
from sim.env import MjlabEnv


class SimRenderer:
    def __init__(self, simulation: MjlabEnv, *, jpeg_quality: int) -> None:
        self.simulation = simulation
        self.jpeg_quality = jpeg_quality

    def capture_jpeg(self) -> bytes:
        image = self.simulation.render()
        return self._encode(image)

    def capture_rgbd(self) -> tuple[bytes, ProjectionContext]:
        image, projection = self.simulation.render_rgbd()
        return self._encode(image), projection

    def capture_depth(self) -> ProjectionContext:
        return self.simulation.render_depth()

    def capture_demo_rgb(self) -> np.ndarray:
        """Capture the same offscreen camera view that is sent to the VLM."""
        return self.simulation.render_demo_rgb()

    def _encode(self, image: np.ndarray) -> bytes:
        buffer = BytesIO()
        iio.imwrite(
            buffer,
            image,
            extension=".jpg",
            quality=self.jpeg_quality,
        )
        return buffer.getvalue()
