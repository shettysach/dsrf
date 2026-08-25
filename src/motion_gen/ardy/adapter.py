"""ARDY implementation of the motion-generation interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
