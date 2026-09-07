from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pyarrow as pa

from shared.messages import (
    AgentCommand,
    ContactGoal,
    EndEffectorSelection,
    EndEffectorTarget,
    GroundingRequest,
    GroundingResult,
    PipelineError,
    RootPathGoal,
    VisualObservation,
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
    if command.reasoning is not None:
        metadata["reasoning"] = command.reasoning
    if command.terminal:
        metadata["terminal"] = "true"
    if command.contact_goal is not None:
        metadata["contact_goal"] = json.dumps(asdict(command.contact_goal))
    if command.root_path_goal is not None:
        metadata["root_path_goal"] = json.dumps(asdict(command.root_path_goal))
    return pa.array([command.text], type=pa.string()), metadata


def agent_command_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> AgentCommand:
    return AgentCommand(
        observation_id=_observation_id(metadata),
        text=_string_from_arrow(value),
        motion=str(metadata["motion"]),
        target_xys=_target_xys(metadata),
        direction=_direction(metadata),
        end_effectors=_end_effectors(metadata),
        reasoning=(str(metadata["reasoning"]) if "reasoning" in metadata else None),
        terminal=metadata.get("terminal") == "true",
        contact_goal=_contact_goal(metadata),
        root_path_goal=_root_path_goal(metadata),
    )


def _contact_goal(metadata: dict[str, Any]) -> ContactGoal | None:
    if "contact_goal" not in metadata:
        return None
    value = json.loads(metadata["contact_goal"])
    return ContactGoal(
        body=value["body"],
        target_xy=tuple(value["target_xy"]),
        points=tuple(
            EndEffectorTarget(
                p["name"],
                tuple(p["target_xyz"]),
                tuple(p["palm_normal"]) if p.get("palm_normal") is not None else None,
            )
            for p in value["points"]
        ),
        goal_half_size=value["goal_half_size"],
        maintain_contact=bool(value.get("maintain_contact", False)),
    )


def _root_path_goal(metadata: dict[str, Any]) -> RootPathGoal | None:
    if "root_path_goal" not in metadata:
        return None
    value = json.loads(metadata["root_path_goal"])
    return RootPathGoal(
        approach_xy=tuple(value["approach_xy"]),
        target_xy=tuple(value["target_xy"]),
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
        "palm_normals": json.dumps(
            [target.palm_normal for target in result.end_effectors],
            separators=(",", ":"),
        ),
    }


def grounding_result_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> GroundingResult:
    target = value.to_pylist()
    waypoint_count = _waypoint_count(metadata)
    end_effector_names = _end_effector_names(metadata)
    palm_normals = _palm_normals(metadata, len(end_effector_names))
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
                palm_normal,
            )
            for name, palm_normal, index in zip(
                end_effector_names,
                palm_normals,
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


def _palm_normals(
    metadata: dict[str, Any], expected_count: int
) -> tuple[tuple[float, float, float] | None, ...]:
    """Read optional normals while accepting messages produced before this field."""
    if "palm_normals" not in metadata:
        return (None,) * expected_count
    values = json.loads(str(metadata["palm_normals"]))
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError("palm_normals metadata must match end_effectors")
    try:
        return tuple(None if value is None else _target_xyz(value) for value in values)
    except (IndexError, TypeError) as exc:
        raise ValueError("palm_normals metadata is invalid") from exc


def _end_effectors_json(end_effectors: tuple[EndEffectorTarget, ...]) -> str:
    return json.dumps(
        [
            {
                "name": target.name,
                "target_xyz": target.target_xyz,
                "palm_normal": target.palm_normal,
            }
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
                (
                    _target_xyz(value["palm_normal"])
                    if value.get("palm_normal") is not None
                    else None
                ),
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
