from __future__ import annotations

import json
from dataclasses import dataclass

KINEMATIC_PLANNER_TOOL = {
    "type": "function",
    "function": {
        "name": "kinematic_planner_command",
        "description": "Choose the robot's next kinematic-planner command.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["motion"],
            "properties": {
                "motion": {"type": "string", "enum": ["stand", "walk"]},
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right"],
                },
                "waypoints_2d": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True)
class KinematicPlannerCommand:
    motion: str
    direction: str | None
    waypoints_2d: tuple[tuple[int, int], ...]


def parse_kinematic_planner_command(text: str) -> KinematicPlannerCommand:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command must be a JSON object") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("motion"), str):
        raise ValueError("Planner command must include motion")
    if not set(payload) <= {"motion", "direction", "waypoints_2d"}:
        raise ValueError("Planner command has unsupported fields")

    motion = payload["motion"]
    direction = payload.get("direction")
    waypoints = payload.get("waypoints_2d", [])
    if not isinstance(waypoints, list):
        raise ValueError("waypoints_2d must be a list")
    parsed_waypoints: list[tuple[int, int]] = []
    for waypoint in waypoints:
        match waypoint:
            case [int(x), int(y)] if type(x) is int and type(y) is int:
                parsed_waypoints.append((x, y))
            case _:
                raise ValueError(
                    "Each waypoint must be [x, y] with integer coordinates"
                )

    if motion == "stand" and direction in {None, "forward"} and not parsed_waypoints:
        return KinematicPlannerCommand("stand", None, ())
    if motion == "walk" and isinstance(direction, str) and not parsed_waypoints:
        if direction in {"forward", "backward", "left", "right"}:
            return KinematicPlannerCommand("walk", direction, ())
    if motion == "walk" and direction is None and parsed_waypoints:
        return KinematicPlannerCommand("walk", None, tuple(parsed_waypoints))
    raise ValueError("walk requires exactly one of direction or waypoints_2d")
