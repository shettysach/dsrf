from __future__ import annotations

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
