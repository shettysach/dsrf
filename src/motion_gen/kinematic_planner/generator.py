from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from motion_gen.kinematic_planner.conditioning import (
    PLANNER_CONTEXT_FRAMES,
    build_planner_inputs,
)
from shared.g1 import standing_qpos
from shared.onnx import create_onnx_session


class KinematicPlanner:
    """Text-to-motion wrapper for NVIDIA's G1 kinematic planner."""

    fps = 30

    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.session = create_onnx_session(model_path, device=self.device)
        initial = standing_qpos()
        self._context = np.tile(initial, (1, PLANNER_CONTEXT_FRAMES, 1))

    def generate(
        self,
        motion: str,
        target_xy: tuple[float, float] | None,
        direction: str | None = None,
    ) -> np.ndarray:
        inputs = build_planner_inputs(self._context, motion, target_xy, direction)
        outputs = self.session.run(None, inputs)
        padded_qpos = np.asarray(outputs[0], dtype=np.float32)
        frame_count = int(np.asarray(outputs[1]).reshape(-1)[0])
        qpos = np.ascontiguousarray(padded_qpos[0, :frame_count])
        self._context = qpos[-PLANNER_CONTEXT_FRAMES:][None].copy()
        return qpos
