from __future__ import annotations

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


def planner_mode(motion: str) -> PlannerMode:
    if motion == "stand":
        return PlannerMode.IDLE
    if motion in {"walk", "turn"}:
        return PlannerMode.WALK
    raise ValueError(
        f"Unsupported motion {motion!r}; expected 'stand', 'walk', or 'turn'"
    )


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


def planner_turn_direction(direction: str) -> tuple[float, float]:
    if direction == "left":
        return (0.0, 1.0)
    if direction == "right":
        return (0.0, -1.0)
    raise ValueError(f"Unsupported turn direction {direction!r}; expected left or right")
