"""ARDY implementation of the motion-generation interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from motion_gen.targets import TimedTargets
from shared.messages import AgentCommand

if TYPE_CHECKING:
    from motion_gen.ardy.generator import Ardy
    from motion_gen.ardy.text_encoder import TextEncoder


class ArdyMotionGenerator:
    """Encode commands and invoke ARDY through the common generator contract."""

    def __init__(self, generator: Ardy, text_encoder: TextEncoder) -> None:
        self._generator = generator
        self._text_encoder = text_encoder
        self.fps: float = float(generator.fps)
        # All released G1 checkpoints used here generate 52 frames. Keeping a
        # fallback also preserves the lightweight backend doubles used by the
        # common motion-generator contract tests.
        self.window_frames = int(
            getattr(getattr(generator, "model", None), "gen_horizon_len", 52)
        )
        self._embeddings: dict[str, torch.Tensor] = {}

    def generate_window(
        self, motion: str, samples: tuple[TimedTargets, ...], history: np.ndarray
    ) -> torch.Tensor:
        self._generator.observe(history)
        if motion not in self._embeddings:
            self._embeddings[motion] = self._text_encoder.encode(motion)
        return self._generator.generate(self._embeddings[motion], (), samples=samples)

    def generate(self, command: AgentCommand) -> torch.Tensor:
        if command.direction is not None:
            raise ValueError(
                "Directional commands are only supported by kinematic_planner"
            )
        embedding = self._text_encoder.encode(command.motion)
        return self._generator.generate(
            embedding,
            command.target_xys,
            command.end_effectors,
        )
