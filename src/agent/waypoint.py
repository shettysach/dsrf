from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class WaypointCommand:
    motion: str
    waypoints_2d: tuple[tuple[int, int], ...]


def parse_waypoint_command(text: str) -> WaypointCommand:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command is not valid JSON") from exc

    match payload:
        case {"motion": str(motion), "waypoints_2d": list(waypoints)} if motion.strip():
            return WaypointCommand(
                motion.strip(), tuple(_waypoint(waypoint) for waypoint in waypoints)
            )
        case _:
            raise ValueError("Expected {motion: <text>, waypoints_2d: [[x, y], ...]}")


def _waypoint(value: object) -> tuple[int, int]:
    match value:
        case [int(x), int(y)] if type(x) is int and type(y) is int:
            return (x, y)
        case _:
            raise ValueError("Each waypoint must be [x, y] with integer coordinates")
