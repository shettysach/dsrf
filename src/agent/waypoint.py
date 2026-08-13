from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class WaypointCommand:
    motion: str
    waypoint_2d: tuple[int, int] | None


def parse_waypoint_command(text: str) -> WaypointCommand:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command is not valid JSON") from exc

    match payload:
        case {"motion": str(motion), "waypoint_2d": None} if motion.strip():
            return WaypointCommand(motion.strip(), None)
        case {"motion": str(motion), "waypoint_2d": [int(x), int(y)]} if motion.strip():
            return WaypointCommand(motion.strip(), (x, y))
        case _:
            raise ValueError("Expected {motion: <text>, waypoint_2d: null | [x, y]}")
