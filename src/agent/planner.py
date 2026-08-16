from __future__ import annotations

import json

KINEMATIC_COMMAND_TOOL_NAME = "submit_kinematic_command"
KINEMATIC_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": KINEMATIC_COMMAND_TOOL_NAME,
        "description": "Submit the robot's next kinematic-planner command.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["motion", "direction"],
            "properties": {
                "motion": {"type": "string", "enum": ["stand", "walk"]},
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right"],
                },
            },
        },
    },
}


def parse_kinematic_command(text: str) -> tuple[str, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command must be a JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"motion", "direction"}:
        raise ValueError("Planner command must contain only motion and direction")
    motion = payload["motion"]
    direction = payload["direction"]
    if motion == "stand" and direction == "forward":
        return "stand", None
    if motion == "walk" and direction in {"forward", "backward", "left", "right"}:
        return "walk", direction
    raise ValueError("Unsupported planner motion or direction")
