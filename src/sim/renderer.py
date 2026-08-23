from __future__ import annotations

import torch
from torchvision.io import encode_jpeg

from sim.camera import ProjectionContext
from sim.env import MjlabEnv


class SimRenderer:
    def __init__(self, simulation: MjlabEnv, *, jpeg_quality: int) -> None:
        self.simulation = simulation
        self.jpeg_quality = jpeg_quality

    def capture_rgbd(self) -> tuple[bytes, ProjectionContext]:
        image, projection = self.simulation.capture_rgbd()
        return self._encode(image), projection

    def _encode(self, image: torch.Tensor) -> bytes:
        encoded = encode_jpeg(image.permute(2, 0, 1), quality=self.jpeg_quality)
        assert isinstance(encoded, torch.Tensor)
        return encoded.cpu().numpy().tobytes()
