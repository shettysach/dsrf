from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class WaypointCommand:
    motion: str
    waypoint_2d: tuple[int, int] | None


def parse_waypoint_command(text: str) -> WaypointCommand:
    if not text.strip():
        raise ValueError("Command is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Command must be a JSON object")
    payload = cast(dict[str, object], payload)
    expected = {"motion", "waypoint_2d"}
    if set(payload) != expected:
        raise ValueError("Command must contain only motion and waypoint_2d")

    motion = payload["motion"]
    waypoint = payload["waypoint_2d"]
    if not isinstance(motion, str):
        raise ValueError("Command motion must be a string")
    motion = motion.strip()
    if not motion:
        raise ValueError("Command motion must not be empty")
    if waypoint is None:
        return WaypointCommand(motion=motion, waypoint_2d=None)

    if not isinstance(waypoint, list) or len(waypoint) != 2:
        raise ValueError("waypoint_2d must be null or [x,y]")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in waypoint):
        raise ValueError("Waypoint coordinates must be integers")
    x, y = cast(list[int], waypoint)
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError("Waypoint coordinates must be in [0,1000]")
    return WaypointCommand(motion=motion, waypoint_2d=(x, y))
