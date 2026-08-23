"""Backend-neutral motion generation contract."""

from __future__ import annotations

from typing import Protocol

import torch

from shared.messages import AgentCommand


class MotionGenerator(Protocol):
    """Generate source-rate G1 trajectories from an agent command."""

    fps: float
    last_encode_ms: float | None

    def generate(self, command: AgentCommand) -> torch.Tensor: ...
