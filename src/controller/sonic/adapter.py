"""SONIC implementation of the generic controller contract."""

from __future__ import annotations

from pathlib import Path

import torch

from controller import RobotState
from controller.sonic.policy import MotionReference, SonicPolicy
from shared.messages import MotionChunk

__all__ = ["SonicController"]


class SonicController:
    """Adapt SONIC's policy API to the generic simulation controller contract."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        device: str = "cpu",
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.policy = SonicPolicy(
            bundle_dir,
            device=device,
            cuda_stream=cuda_stream,
        )

    @property
    def reference(self) -> MotionReference:
        """Expose SONIC's optional debug reference without leaking it to runtime."""
        return self.policy.reference

    def load_motion(self, chunk: MotionChunk, state: RobotState) -> None:
        self.policy.load_motion(chunk, state.root_pos_w, state.root_quat_w)

    def act(self, state: RobotState) -> tuple[torch.Tensor, bool]:
        return self.policy.infer(state)
