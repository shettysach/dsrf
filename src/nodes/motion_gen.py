from __future__ import annotations

import time
from typing import Any

from dora import Node

from motion_gen.kinematic_planner import KinematicPlanner
from motion_gen.resample import resample_motion
from shared.config import ArdyConfig, KinematicPlannerConfig, MotionGenConfig
from shared.messages import (
    SONIC_FPS,
    AgentCommand,
    EncodedCommand,
    PipelineError,
    agent_command_from_arrow,
    encoded_command_from_arrow,
    motion_to_arrow,
    pipeline_error_to_arrow,
)


def _create_generator(cfg: MotionGenConfig) -> Any:
    match cfg.backend:
        case ArdyConfig():
            from motion_gen.ardy.generator import Ardy

            return Ardy(cfg.backend.checkpoints_dir, device=cfg.device)

        case KinematicPlannerConfig():
            return KinematicPlanner(cfg.backend.planner_onnx, device=cfg.device)


def main() -> None:
    cfg = MotionGenConfig.from_env()
    node = Node()
    generator = _create_generator(cfg)
    ardy_backend = isinstance(cfg.backend, ArdyConfig)
    input_id = "encoded_command" if ardy_backend else "command"

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] != input_id:
            continue

        metadata = dict(event.get("metadata") or {})
        request: AgentCommand | EncodedCommand
        if ardy_backend:
            request = encoded_command_from_arrow(event["value"], metadata)
        else:
            request = agent_command_from_arrow(event["value"], metadata)
        started_at = time.perf_counter()
        try:
            try:
                if isinstance(request, EncodedCommand):
                    source_qpos = generator.generate(
                        request.embedding,
                        request.target_xy,
                    )
                else:
                    source_qpos = generator.generate(
                        request.motion,
                        request.target_xy,
                        request.direction,
                    )
            except ValueError as exc:
                node.log(
                    "warn",
                    f"[OBS {request.observation_id}] invalid command: "
                    f"{request.text!r} error={str(exc)!r}",
                    target="dsrf.motion_gen",
                    fields={
                        "event": "invalid_command",
                        "observation_id": str(request.observation_id),
                        "command": request.text,
                        "detail": str(exc),
                    },
                )
                error = PipelineError("motion-gen", request.observation_id, str(exc))
                node.send_output("error", pipeline_error_to_arrow(error))
                continue

            chunk = resample_motion(
                source_qpos,
                source_fps=generator.fps,
                observation_id=request.observation_id,
                command=request.text,
            )
        except Exception as exc:
            plan_ms = (time.perf_counter() - started_at) * 1000.0
            detail = f"{type(exc).__name__}: {exc}"
            node.log(
                "error",
                f"[OBS {request.observation_id}] motion generation failed: {detail}",
                target="dsrf.motion_gen",
                fields={
                    "event": "motion_generation_error",
                    "observation_id": str(request.observation_id),
                    "command": request.text,
                    "plan_ms": f"{plan_ms:.1f}",
                    "detail": detail,
                },
            )
            raise

        plan_ms = (time.perf_counter() - started_at) * 1000.0
        output_frames = len(chunk.qpos)
        duration_s = output_frames / SONIC_FPS
        node.log(
            "info",
            f"[OBS {request.observation_id}] motion generated: "
            f"command={request.text!r} frames={output_frames} "
            f"duration_s={duration_s:.2f} plan_ms={plan_ms:.1f}",
            target="dsrf.motion_gen",
            fields={
                "event": "motion_generated",
                "observation_id": str(request.observation_id),
                "command": request.text,
                "plan_ms": f"{plan_ms:.1f}",
                "source_frames": str(len(source_qpos)),
                "output_frames": str(output_frames),
                "duration_s": f"{duration_s:.2f}",
            },
        )
        data, motion_metadata = motion_to_arrow(chunk)
        node.send_output("motion", data, metadata=motion_metadata)


if __name__ == "__main__":
    main()
