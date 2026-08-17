from __future__ import annotations

import json
from dataclasses import dataclass

from shared.messages import END_EFFECTOR_NAMES, EndEffectorSelection


@dataclass(frozen=True)
class ArdyCommand:
    motion: str
    waypoints_2d: tuple[tuple[int, int], ...]
    end_effectors: tuple[EndEffectorSelection, ...]


def parse_ardy_command(text: str) -> ArdyCommand:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command is not valid JSON") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("motion"), str):
        raise ValueError("Command must include motion")

    waypoints = payload.get("waypoints_2d", [])
    end_effectors = payload.get("end_effectors", [])
    if not isinstance(waypoints, list) or not isinstance(end_effectors, list):
        raise ValueError("Constraints must be lists")

    parsed_waypoints = tuple(map(_point, waypoints))
    parsed_end_effectors = tuple(map(_end_effector, end_effectors))

    names = [e.name for e in parsed_end_effectors]
    if len(names) != len(set(names)):
        raise ValueError("Each end effector may be constrained once")

    return ArdyCommand(payload["motion"], parsed_waypoints, parsed_end_effectors)


def _point(value: object) -> tuple[int, int]:
    match value:
        case [int(x), int(y)] if type(x) is int and type(y) is int:
            return (x, y)
        case _:
            raise ValueError(
                "Each image target must be [x, y] with integer coordinates"
            )


def _end_effector(value: object) -> EndEffectorSelection:
    match value:
        case {"name": str(name), "target_2d": target} if name in END_EFFECTOR_NAMES:
            return EndEffectorSelection(name, _point(target))
        case _:
            supported = ", ".join(sorted(END_EFFECTOR_NAMES))
            raise ValueError(
                "Each end effector requires a supported name "
                f"({supported}) and target_2d"
            )
