"""SONIC implementation of the generic controller contract."""

from __future__ import annotations

from pathlib import Path

import torch

from controller import ControlOutput, RobotState
from controller.g1_command import G1CommandTransform
from controller.reference import MotionReference
from controller.sonic.policy import SonicPolicy
from shared.messages import MotionChunk

__all__ = ["SonicController"]


class SonicController:
    """Adapt SONIC's policy API to the generic simulation controller contract."""

    def __init__(
        self,
        bundle_dir: Path,
        command_transform: G1CommandTransform,
        *,
        device: str = "cpu",
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.command_transform = command_transform
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

    def act(self, state: RobotState) -> ControlOutput:
        raw_action, completed = self.policy.infer(state)
        return ControlOutput(
            joint_target=self.command_transform.decode(raw_action.squeeze(0)),
            completed=completed,
        )
