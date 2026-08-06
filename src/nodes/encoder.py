from __future__ import annotations

import time

from dora import Node

from encoder import TextEncoder
from shared.config import TextEncoderConfig
from shared.messages import (
    EncodedCommand,
    PipelineError,
    agent_command_from_arrow,
    encoded_command_to_arrow,
    pipeline_error_to_arrow,
)


def main() -> None:
    cfg = TextEncoderConfig.from_env()
    node = Node()
    encoder = TextEncoder(cfg.model, device=cfg.device)

    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "command":
            continue

        metadata = dict(event.get("metadata") or {})
        request = agent_command_from_arrow(event["value"], metadata)
        started_at = time.perf_counter()
        try:
            if request.direction is not None:
                raise ValueError(
                    "Directional commands are only supported by kinematic_planner"
                )
            embedding = encoder.encode(request.motion)
        except ValueError as exc:
            node.log(
                "warn",
                f"[OBS {request.observation_id}] invalid command: "
                f"{request.text!r} error={str(exc)!r}",
                target="dsrf.text_encoder",
                fields={
                    "event": "invalid_command",
                    "observation_id": str(request.observation_id),
                    "command": request.text,
                    "detail": str(exc),
                },
            )
            error = PipelineError("text-encoder", request.observation_id, str(exc))
            node.send_output("error", pipeline_error_to_arrow(error))
            continue
        encoded = EncodedCommand(
            observation_id=request.observation_id,
            text=request.text,
            motion=request.motion,
            target_xy=request.target_xy,
            embedding=embedding,
        )
        value, output_metadata = encoded_command_to_arrow(encoded)
        node.send_output("encoded_command", value, metadata=output_metadata)

        encode_ms = (time.perf_counter() - started_at) * 1000.0
        node.log(
            "info",
            f"[OBS {request.observation_id}] command encoded: "
            f"text={request.motion!r} encode_ms={encode_ms:.1f}",
            target="dsrf.text_encoder",
            fields={
                "event": "command_encoded",
                "observation_id": str(request.observation_id),
                "command": request.motion,
                "encode_ms": f"{encode_ms:.1f}",
            },
        )


if __name__ == "__main__":
    main()
