from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from motion_gen.kinematic_planner.conditioning import (
    PLANNER_CONTEXT_FRAMES,
    build_planner_inputs,
)
from shared.g1 import standing_qpos
from shared.geometry import local_xy_to_world, world_xy_to_local, yaw_from_quat_wxyz
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
        target_xys: tuple[tuple[float, float], ...],
        direction: str | None = None,
    ) -> np.ndarray:
        if not target_xys:
            return self._generate_once(motion, None, direction)
        if direction is not None:
            raise ValueError("Motion command cannot have both a waypoint and direction")

        # All resolved targets are robot-local to the frozen observation. Convert
        # each once to a world target, then re-express it in the current root frame
        # before each short planner invocation.
        initial_root = self._context[0, -1]
        initial_yaw = yaw_from_quat_wxyz(initial_root[3:7])
        world_targets = tuple(
            initial_root[:2] + local_xy_to_world(forward, left, initial_yaw)
            for forward, left in target_xys
        )
        segments = []
        for world_target in world_targets:
            root = self._context[0, -1]
            yaw = yaw_from_quat_wxyz(root[3:7])
            target_xy = world_xy_to_local(world_target - root[:2], yaw)
            segments.append(self._generate_once(motion, tuple(target_xy), None))
        return np.ascontiguousarray(np.concatenate(segments, axis=0))

    def _generate_once(
        self,
        motion: str,
        target_xy: tuple[float, float] | None,
        direction: str | None,
    ) -> np.ndarray:
        inputs = build_planner_inputs(self._context, motion, target_xy, direction)
        outputs = self.session.run(None, inputs)
        padded_qpos = np.asarray(outputs[0], dtype=np.float32)
        frame_count = int(np.asarray(outputs[1]).reshape(-1)[0])
        qpos = np.ascontiguousarray(padded_qpos[0, :frame_count])
        self._context = qpos[-PLANNER_CONTEXT_FRAMES:][None].copy()
        return qpos
