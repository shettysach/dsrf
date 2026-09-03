from __future__ import annotations

import numpy as np
import torch
from torchvision.io import encode_jpeg

from sim.camera import ProjectionContext
from sim.env import MjlabEnv


class SimRenderer:
    def __init__(self, simulation: MjlabEnv, *, jpeg_quality: int) -> None:
        self.simulation = simulation
        self.jpeg_quality = jpeg_quality

    def capture_rgbd(self) -> tuple[bytes, ProjectionContext]:
        jpeg, projection, _ = self.capture_observation()
        return jpeg, projection

    def capture_observation(self) -> tuple[bytes, ProjectionContext, np.ndarray]:
        """Capture the image sent to the VLM, retaining its RGB pixels for demos."""
        image, projection = self.simulation.capture_rgbd()
        # Keep an immutable snapshot while inference and motion generation run.
        # Encoding on the CPU also avoids torchvision's unreliable CUDA JPEG path
        # on machines whose display server lacks NVIDIA GLX support.
        image_cpu = image.detach().cpu()
        rgb = image_cpu.numpy().copy()
        return self._encode(image_cpu), projection, rgb

    def capture_demo_rgb(self) -> np.ndarray:
        """Capture an RGB frame from the VLM observation camera for a demo video."""
        image, _ = self.simulation.capture_rgbd()
        return image.detach().cpu().numpy()

    def _encode(self, image: torch.Tensor) -> bytes:
        encoded = encode_jpeg(image.permute(2, 0, 1), quality=self.jpeg_quality)
        assert isinstance(encoded, torch.Tensor)
        return encoded.cpu().numpy().tobytes()
