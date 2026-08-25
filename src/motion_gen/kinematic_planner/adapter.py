"""Node-facing adapter for the kinematic planner."""

from __future__ import annotations

import torch

from motion_gen.kinematic_planner.generator import KinematicPlanner
from shared.messages import AgentCommand


class KinematicPlannerMotionGenerator:
    """Invoke the planner through the common generator contract."""

    def __init__(self, generator: KinematicPlanner) -> None:
        self._generator = generator
        self.fps: float = float(generator.fps)

    def generate(self, command: AgentCommand) -> torch.Tensor:
        if command.end_effectors:
            raise ValueError("End-effector constraints are only supported by ARDY")
        return self._generator.generate(
            command.motion,
            command.target_xys,
            command.direction,
        )
