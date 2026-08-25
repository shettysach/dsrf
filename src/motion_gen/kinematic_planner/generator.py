from __future__ import annotations

import math
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply_yaw, yaw_quat

from motion_gen.kinematic_planner.conditioning import (
    PLANNER_CONTEXT_FRAMES,
    build_planner_inputs,
)
from shared.g1 import standing_qpos
from shared.onnx import StaticOnnxModel


class KinematicPlanner:
    """Text-to-motion wrapper for NVIDIA's G1 kinematic planner."""

    fps = 30

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = StaticOnnxModel(
            model_path,
            device=self.device,
            cuda_stream=cuda_stream,
        )
        initial = torch.as_tensor(
            standing_qpos(), dtype=torch.float32, device=self.device
        )
        self._context = initial.repeat(1, PLANNER_CONTEXT_FRAMES, 1)

    def generate(
        self,
        motion: str,
        target_xys: tuple[tuple[float, float], ...],
        direction: str | None = None,
    ) -> torch.Tensor:
        if not target_xys:
            return self._generate_once(motion, None, direction)
        if direction is not None:
            raise ValueError("Motion command cannot have both a waypoint and direction")
        if any(math.hypot(*target) <= 1e-6 for target in target_xys):
            raise ValueError("walk target_xy must be non-zero")

        # All resolved targets are robot-local to the frozen observation. Convert
        # each once to a world target, then re-express it in the current root frame
        # before each short planner invocation.
        initial_root = self._context[0, -1]
        world_targets = tuple(
            initial_root[:2]
            + quat_apply_yaw(
                yaw_quat(initial_root[3:7]),
                initial_root.new_tensor((forward, left, 0.0)),
            )[:2]
            for forward, left in target_xys
        )
        segments = []
        for world_target in world_targets:
            root = self._context[0, -1]
            delta_w = torch.cat((world_target - root[:2], root.new_zeros(1)))
            target_xy = quat_apply_inverse(yaw_quat(root[3:7]), delta_w)[:2]
            segments.append(
                self._generate_once(
                    motion,
                    target_xy,
                    None,
                )
            )
        return torch.cat(segments).contiguous()

    def _generate_once(
        self,
        motion: str,
        target_xy: tuple[float, float] | torch.Tensor | None,
        direction: str | None,
    ) -> torch.Tensor:
        inputs = build_planner_inputs(self._context, motion, target_xy, direction)
        outputs = self.model.run(inputs)
        padded_qpos = outputs["mujoco_qpos"]
        frame_count = int(outputs["num_pred_frames"].item())
        qpos = padded_qpos[0, :frame_count].clone()
        self._context = qpos[-PLANNER_CONTEXT_FRAMES:][None].clone()
        return qpos
