"""Construction of configured motion-generation backends."""

from __future__ import annotations

import torch

from motion_gen.ardy.adapter import ArdyMotionGenerator
from motion_gen.generator import MotionGenerator
from motion_gen.kinematic_planner.adapter import KinematicPlannerMotionGenerator
from shared.config import ArdyConfig, KinematicPlannerConfig, MotionGenConfig


def create_motion_generator(
    cfg: MotionGenConfig,
    *,
    cuda_stream: torch.cuda.Stream | None = None,
) -> MotionGenerator:
    match cfg.backend:
        case ArdyConfig():
            from motion_gen.ardy.generator import Ardy
            from motion_gen.ardy.text_encoder import TextEncoder

            return ArdyMotionGenerator(
                Ardy(
                    cfg.backend.checkpoints_dir,
                    device=cfg.device,
                    constraint_cfg_weight=cfg.backend.constraint_cfg_weight,
                    seed=cfg.backend.seed,
                    end_effector_diagnostics=cfg.backend.end_effector_diagnostics,
                ),
                TextEncoder(
                    cfg.backend.text_encoder_model,
                    device=cfg.backend.text_encoder_device,
                ),
            )
        case KinematicPlannerConfig():
            from motion_gen.kinematic_planner.generator import KinematicPlanner

            return KinematicPlannerMotionGenerator(
                KinematicPlanner(
                    cfg.backend.planner_onnx,
                    device=cfg.device,
                    cuda_stream=cuda_stream,
                )
            )
