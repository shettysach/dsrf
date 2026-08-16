from __future__ import annotations

import json
from dataclasses import dataclass

from shared.messages import END_EFFECTOR_NAMES, EndEffectorSelection

_FIELDS = {"motion", "waypoints_2d", "end_effectors"}


@dataclass(frozen=True)
class ConstraintCommand:
    motion: str
    waypoints_2d: tuple[tuple[int, int], ...]
    end_effectors: tuple[EndEffectorSelection, ...]


def parse_constraint_command(text: str) -> ConstraintCommand:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command is not valid JSON") from exc

    match payload:
        case {"motion": str(motion)} if motion.strip() and payload.keys() <= _FIELDS:
            match (
                payload.get("waypoints_2d", []),
                payload.get("end_effectors", []),
            ):
                case (list(waypoints), list(end_effectors)):
                    parsed_waypoints = tuple(_point(waypoint) for waypoint in waypoints)
                    parsed_end_effectors = tuple(
                        _end_effector(end_effector) for end_effector in end_effectors
                    )
                    if parsed_waypoints and parsed_end_effectors:
                        raise ValueError(
                            "A command cannot combine waypoints and end effectors"
                        )
                    names = [end_effector.name for end_effector in parsed_end_effectors]
                    if len(names) != len(set(names)):
                        raise ValueError("Each end effector may be constrained once")
                    return ConstraintCommand(
                        motion.strip(), parsed_waypoints, parsed_end_effectors
                    )
                case _:
                    raise ValueError(
                        "waypoints_2d and end_effectors must be lists when provided"
                    )
        case _:
            raise ValueError(
                "Expected {motion: <text>, waypoints_2d?: [[x, y], ...], "
                "end_effectors?: [{name, target_2d}, ...]}"
            )


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
