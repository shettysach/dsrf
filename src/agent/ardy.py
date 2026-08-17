from __future__ import annotations

from shared.messages import END_EFFECTOR_NAMES

ARDY_TOOL = {
    "type": "function",
    "function": {
        "name": "ardy_command",
        "description": "Choose ARDY's next motion and optional image constraints.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["motion"],
            "properties": {
                "motion": {"type": "string"},
                "waypoints_2d": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "end_effectors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "target_2d"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": sorted(END_EFFECTOR_NAMES),
                            },
                            "target_2d": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1000,
                                },
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                    },
                },
            },
        },
    },
}
