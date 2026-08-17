from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum


class PlannerMode(IntEnum):
    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    SQUAT = 4
    KNEEL_TWO_LEG = 5
    KNEEL_ONE_LEG = 6
    LYING_FACEDOWN = 7
    HAND_CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_JAB = 11
    RIGHT_JAB = 12
    RANDOM_PUNCHES = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    HAPPY = 17
    STEALTH = 18
    INJURED = 19
    CAREFUL = 20
    OBJECT_CARRYING = 21
    CROUCH = 22
    HAPPY_DANCE = 23
    ZOMBIE = 24
    POINT = 25
    SCARED = 26


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

    if motion == "stand" and not parsed_waypoints:
        if direction is None or direction in {"forward", "backward", "left", "right"}:
            return KinematicPlannerCommand("stand", direction, ())
    if motion == "walk" and isinstance(direction, str) and not parsed_waypoints:
        if direction in {"forward", "backward", "left", "right"}:
            return KinematicPlannerCommand("walk", direction, ())
    if motion == "walk" and direction is None and parsed_waypoints:
        return KinematicPlannerCommand("walk", None, tuple(parsed_waypoints))
    raise ValueError("walk requires exactly one of direction or waypoints_2d")


def planner_mode(motion: str) -> PlannerMode:
    if motion == "stand":
        return PlannerMode.IDLE
    if motion == "walk":
        return PlannerMode.WALK
    raise ValueError(f"Unsupported motion {motion!r}; expected 'stand' or 'walk'")


def planner_direction(direction: str) -> tuple[float, float]:
    if direction == "forward":
        return (1.0, 0.0)
    if direction == "right":
        return (0.0, -1.0)
    if direction == "left":
        return (0.0, 1.0)
    if direction == "backward":
        return (-1.0, 0.0)
    raise ValueError(
        f"Unsupported direction {direction!r}; expected forward, backward, left, or right"
    )
