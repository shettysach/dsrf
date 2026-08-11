from __future__ import annotations

import time
from typing import Any

import numpy as np
from dora import Node

from motion_gen.kinematic_planner import KinematicPlanner
from motion_gen.resample import resample_motion
from shared.arrow import (
    agent_command_from_arrow,
    motion_to_arrow,
    pipeline_error_to_arrow,
)
from shared.config import ArdyConfig, KinematicPlannerConfig, MotionGenConfig
from shared.messages import SONIC_FPS, AgentCommand, MotionChunk, PipelineError


def _create_generator(cfg: MotionGenConfig) -> Any:
    match cfg.backend:
        case ArdyConfig():
            from motion_gen.ardy.generator import Ardy

            return Ardy(cfg.backend.checkpoints_dir, device=cfg.device)

        case KinematicPlannerConfig():
            return KinematicPlanner(cfg.backend.planner_onnx, device=cfg.device)


def _create_text_encoder(cfg: MotionGenConfig) -> Any | None:
    if not isinstance(cfg.backend, ArdyConfig):
        return None

    from motion_gen.ardy.text_encoder import TextEncoder

    return TextEncoder(
        cfg.backend.text_encoder_model,
        device=cfg.backend.text_encoder_device,
    )


def main() -> None:
    cfg = MotionGenConfig.from_env()
    node = Node()
    generator = _create_generator(cfg)
    text_encoder = _create_text_encoder(cfg)

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "command":
            continue

        metadata = dict(event.get("metadata") or {})
        request = agent_command_from_arrow(event["value"], metadata)
        started_at = time.perf_counter()
        encode_ms: float | None = None
        try:
            if text_encoder is None:
                source_qpos = generator.generate(
                    request.motion,
                    request.target_xy,
                    request.direction,
                )
            else:
                if request.direction is not None:
                    raise ValueError(
                        "Directional commands are only supported by kinematic_planner"
                    )

                encode_started_at = time.perf_counter()
                embedding = text_encoder.encode(request.motion)
                encode_ms = (time.perf_counter() - encode_started_at) * 1000.0
                source_qpos = generator.generate(embedding, request.target_xy)

            chunk = resample_motion(
                source_qpos,
                source_fps=generator.fps,
                observation_id=request.observation_id,
                command=request.text,
            )
        except ValueError as exc:
            _report_invalid_command(node, request, exc)
            continue
        except Exception as exc:
            _log_generation_error(node, request, started_at, exc)
            raise

        plan_ms = (time.perf_counter() - started_at) * 1000.0
        _log_motion_generated(
            node,
            request,
            source_qpos,
            chunk,
            plan_ms=plan_ms,
            encode_ms=encode_ms,
        )
        data, motion_metadata = motion_to_arrow(chunk)
        node.send_output("motion", data, metadata=motion_metadata)


if __name__ == "__main__":
    main()


def _report_invalid_command(
    node: Node,
    request: AgentCommand,
    error: ValueError,
) -> None:
    detail = str(error)
    node.log(
        "warn",
        f"[OBS {request.observation_id}] invalid command: "
        f"{request.text!r} error={detail!r}",
        target="dsrf.motion_gen",
        fields={
            "event": "invalid_command",
            "observation_id": str(request.observation_id),
            "command": request.text,
            "detail": detail,
        },
    )
    pipeline_error = PipelineError("motion-gen", request.observation_id, detail)
    node.send_output("error", pipeline_error_to_arrow(pipeline_error))


def _log_generation_error(
    node: Node,
    request: AgentCommand,
    started_at: float,
    error: Exception,
) -> None:
    plan_ms = (time.perf_counter() - started_at) * 1000.0
    detail = f"{type(error).__name__}: {error}"
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


def _log_motion_generated(
    node: Node,
    request: AgentCommand,
    source_qpos: np.ndarray,
    chunk: MotionChunk,
    *,
    plan_ms: float,
    encode_ms: float | None,
) -> None:
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
            **({"encode_ms": f"{encode_ms:.1f}"} if encode_ms is not None else {}),
        },
    )
