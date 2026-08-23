from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pyarrow as pa

from shared.messages import (
    MOTION_COLUMNS,
    AgentCommand,
    EndEffectorSelection,
    EndEffectorTarget,
    GroundingRequest,
    GroundingResult,
    MotionChunk,
    PipelineError,
    VisualObservation,
)


def motion_to_arrow(chunk: MotionChunk) -> tuple[pa.Array, dict[str, str]]:
    return pa.array(chunk.qpos.reshape(-1), type=pa.float32()), {
        "observation_id": str(chunk.observation_id),
        "command": chunk.command,
    }


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
    if command.end_effectors:
        metadata["end_effectors"] = _end_effectors_json(command.end_effectors)
    return pa.array([command.text], type=pa.string()), metadata


def agent_command_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> AgentCommand:
    return AgentCommand(
        observation_id=_observation_id(metadata),
        text=_string_from_arrow(value),
        motion=str(metadata["motion"]),
        target_xys=_target_xys(metadata),
        direction=_direction(metadata),
        end_effectors=_end_effectors(metadata),
    )


def observation_to_arrow(
    observation: VisualObservation,
) -> tuple[pa.Array, dict[str, str]]:
    metadata = {
        "observation_id": str(observation.observation_id),
        "mime_type": "image/jpeg",
    }
    if observation.completed_command is not None:
        metadata["completed_command"] = observation.completed_command
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
    )


def grounding_request_to_arrow(
    request: GroundingRequest,
) -> tuple[pa.Array, dict[str, str]]:
    points = request.waypoints_2d + tuple(
        selection.target_2d for selection in request.end_effectors_2d
    )
    return pa.array(
        [coordinate for point in points for coordinate in point],
        type=pa.int32(),
    ), {
        "observation_id": str(request.observation_id),
        "waypoint_count": str(len(request.waypoints_2d)),
        "end_effectors": json.dumps(
            [selection.name for selection in request.end_effectors_2d],
            separators=(",", ":"),
        ),
    }


def grounding_request_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> GroundingRequest:
    waypoint = value.to_pylist()
    waypoint_count = _waypoint_count(metadata)
    end_effector_names = _end_effector_names(metadata)
    if len(waypoint) != (waypoint_count + len(end_effector_names)) * 2:
        raise ValueError(
            "Grounding request payload has the wrong number of coordinates"
        )
    return GroundingRequest(
        observation_id=_observation_id(metadata),
        waypoints_2d=tuple(
            (int(waypoint[index]), int(waypoint[index + 1]))
            for index in range(0, waypoint_count * 2, 2)
        ),
        end_effectors_2d=tuple(
            EndEffectorSelection(name, (int(waypoint[index]), int(waypoint[index + 1])))
            for name, index in zip(
                end_effector_names,
                range(waypoint_count * 2, len(waypoint), 2),
                strict=True,
            )
        ),
    )


def grounding_result_to_arrow(
    result: GroundingResult,
) -> tuple[pa.Array, dict[str, str]]:
    coordinates = [
        coordinate for target in result.target_xys for coordinate in target
    ] + [
        coordinate
        for end_effector in result.end_effectors
        for coordinate in end_effector.target_xyz
    ]
    return pa.array(
        coordinates,
        type=pa.float32(),
    ), {
        "observation_id": str(result.observation_id),
        "waypoint_count": str(len(result.target_xys)),
        "end_effectors": json.dumps(
            [target.name for target in result.end_effectors], separators=(",", ":")
        ),
    }


def grounding_result_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> GroundingResult:
    target = value.to_pylist()
    waypoint_count = _waypoint_count(metadata)
    end_effector_names = _end_effector_names(metadata)
    end_effector_start = waypoint_count * 2
    if len(target) != end_effector_start + len(end_effector_names) * 3:
        raise ValueError("Grounding result payload has the wrong number of coordinates")
    return GroundingResult(
        observation_id=_observation_id(metadata),
        target_xys=tuple(
            (float(target[index]), float(target[index + 1]))
            for index in range(0, end_effector_start, 2)
        ),
        end_effectors=tuple(
            EndEffectorTarget(
                name,
                (
                    float(target[index]),
                    float(target[index + 1]),
                    float(target[index + 2]),
                ),
            )
            for name, index in zip(
                end_effector_names,
                range(end_effector_start, len(target), 3),
                strict=True,
            )
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


def _end_effector_names(metadata: dict[str, Any]) -> tuple[str, ...]:
    values = json.loads(str(metadata.get("end_effectors", "[]")))
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError("end_effectors metadata must contain a list of names")
    return tuple(values)


def _end_effectors_json(end_effectors: tuple[EndEffectorTarget, ...]) -> str:
    return json.dumps(
        [
            {"name": target.name, "target_xyz": target.target_xyz}
            for target in end_effectors
        ],
        separators=(",", ":"),
    )


def _end_effectors(metadata: dict[str, Any]) -> tuple[EndEffectorTarget, ...]:
    if "end_effectors" not in metadata:
        return ()
    values = json.loads(str(metadata["end_effectors"]))
    if not isinstance(values, list):
        raise ValueError("end_effectors metadata must contain a list")
    try:
        return tuple(
            EndEffectorTarget(
                str(value["name"]),
                _target_xyz(value["target_xyz"]),
            )
            for value in values
        )
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("end_effectors metadata is invalid") from exc


def _target_xyz(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError("End-effector target must contain three coordinates")
    return float(value[0]), float(value[1]), float(value[2])


def _direction(metadata: dict[str, Any]) -> str | None:
    if "direction" not in metadata:
        return None
    direction = str(metadata["direction"])
    if direction not in {"forward", "backward", "left", "right"}:
        raise ValueError("Unsupported direction")
    return direction
