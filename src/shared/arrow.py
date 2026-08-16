from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pyarrow as pa

from shared.messages import (
    MOTION_COLUMNS,
    AgentCommand,
    GroundingRequest,
    GroundingResult,
    MotionChunk,
    PipelineError,
    VisualObservation,
)


def motion_to_arrow(chunk: MotionChunk) -> tuple[pa.Array, dict[str, str]]:
    metadata = {
        "observation_id": str(chunk.observation_id),
        "command": chunk.command,
    }
    if chunk.reasoning is not None:
        metadata["reasoning"] = chunk.reasoning
    return pa.array(chunk.qpos.reshape(-1), type=pa.float32()), metadata


def motion_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> MotionChunk:
    flat = np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32)
    if flat.size == 0 or flat.size % MOTION_COLUMNS:
        raise ValueError(
            f"Motion payload has {flat.size} values; expected complete "
            f"{MOTION_COLUMNS}-value frames"
        )
    return MotionChunk(
        observation_id=_observation_id(metadata),
        command=str(metadata["command"]),
        qpos=flat.reshape(-1, MOTION_COLUMNS),
        reasoning=_optional_string(metadata, "reasoning"),
    )


def agent_command_to_arrow(
    command: AgentCommand,
) -> tuple[pa.Array, dict[str, str]]:
    metadata = {
        "observation_id": str(command.observation_id),
        "motion": command.motion,
    }
    if command.target_xys:
        metadata["target_xys"] = json.dumps(command.target_xys, separators=(",", ":"))
    if command.direction is not None:
        metadata["direction"] = command.direction
    if command.reasoning is not None:
        metadata["reasoning"] = command.reasoning
    return pa.array([command.text], type=pa.string()), metadata


def agent_command_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> AgentCommand:
    return AgentCommand(
        observation_id=_observation_id(metadata),
        text=_string_from_arrow(value),
        motion=str(metadata["motion"]),
        target_xys=_target_xys(metadata),
        direction=_direction(metadata),
        reasoning=_optional_string(metadata, "reasoning"),
    )


def observation_to_arrow(
    observation: VisualObservation,
) -> tuple[pa.Array, dict[str, str]]:
    metadata = {
        "observation_id": str(observation.observation_id),
        "run_id": str(observation.run_id),
        "mime_type": "image/jpeg",
    }
    if observation.completed_command is not None:
        metadata["completed_command"] = observation.completed_command
    if observation.collision_detected:
        metadata["collision_detected"] = "true"
    return pa.array([observation.jpeg], type=pa.binary()), metadata


def observation_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> VisualObservation:
    mime_type = metadata.get("mime_type")
    if mime_type != "image/jpeg":
        raise ValueError(f"Unsupported observation MIME type: {mime_type!r}")
    jpeg = _binary_from_arrow(value)
    return VisualObservation(
        observation_id=_observation_id(metadata),
        completed_command=(
            str(metadata["completed_command"])
            if "completed_command" in metadata
            else None
        ),
        jpeg=jpeg,
        run_id=int(metadata.get("run_id", 0)),
        collision_detected=str(metadata.get("collision_detected", "false")).lower()
        == "true",
    )


def grounding_request_to_arrow(
    request: GroundingRequest,
) -> tuple[pa.Array, dict[str, str]]:
    return pa.array(
        [coordinate for waypoint in request.waypoints_2d for coordinate in waypoint],
        type=pa.int32(),
    ), {
        "observation_id": str(request.observation_id),
        "waypoint_count": str(len(request.waypoints_2d)),
    }


def grounding_request_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> GroundingRequest:
    waypoint = value.to_pylist()
    count = _waypoint_count(metadata)
    if len(waypoint) != count * 2:
        raise ValueError(
            "Grounding request payload has the wrong number of coordinates"
        )
    return GroundingRequest(
        observation_id=_observation_id(metadata),
        waypoints_2d=tuple(
            (int(waypoint[index]), int(waypoint[index + 1]))
            for index in range(0, len(waypoint), 2)
        ),
    )


def grounding_result_to_arrow(
    result: GroundingResult,
) -> tuple[pa.Array, dict[str, str]]:
    return pa.array(
        [coordinate for target in result.target_xys for coordinate in target],
        type=pa.float32(),
    ), {
        "observation_id": str(result.observation_id),
        "waypoint_count": str(len(result.target_xys)),
    }


def grounding_result_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> GroundingResult:
    target = value.to_pylist()
    count = _waypoint_count(metadata)
    if len(target) != count * 2:
        raise ValueError("Grounding result payload has the wrong number of coordinates")
    return GroundingResult(
        observation_id=_observation_id(metadata),
        target_xys=tuple(
            (float(target[index]), float(target[index + 1]))
            for index in range(0, len(target), 2)
        ),
    )


def pipeline_error_to_arrow(error: PipelineError) -> pa.Array:
    return _json_to_arrow(asdict(error))


def pipeline_error_from_arrow(value: pa.Array) -> PipelineError:
    data = _json_from_arrow(value)
    return PipelineError(
        source=str(data["source"]),
        observation_id=int(data["observation_id"]),
        detail=str(data["detail"]),
    )


def _json_to_arrow(value: dict[str, Any]) -> pa.Array:
    return pa.array([json.dumps(value, separators=(",", ":"))], type=pa.string())


def _json_from_arrow(value: pa.Array) -> dict[str, Any]:
    decoded = json.loads(_string_from_arrow(value))
    if not isinstance(decoded, dict):
        raise ValueError("Expected a JSON object")
    return decoded


def _string_from_arrow(value: pa.Array) -> str:
    values = value.to_pylist()
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Expected one string")
    return values[0]


def _binary_from_arrow(value: pa.Array) -> bytes:
    values = value.to_pylist()
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise ValueError("Expected one binary value")
    return values[0]


def _observation_id(metadata: dict[str, Any]) -> int:
    return int(metadata["observation_id"])


def _target_xys(metadata: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    if "target_xys" not in metadata:
        return ()
    values = json.loads(str(metadata["target_xys"]))
    if not isinstance(values, list):
        raise ValueError("target_xys metadata must contain a list")
    try:
        return tuple((float(target[0]), float(target[1])) for target in values)
    except (IndexError, TypeError) as exc:
        raise ValueError("target_xys metadata must contain 2D targets") from exc


def _waypoint_count(metadata: dict[str, Any]) -> int:
    return int(metadata["waypoint_count"])


def _direction(metadata: dict[str, Any]) -> str | None:
    if "direction" not in metadata:
        return None
    direction = str(metadata["direction"])
    if direction not in {"forward", "backward", "left", "right"}:
        raise ValueError("Unsupported direction")
    return direction


def _optional_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None
