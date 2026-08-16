from __future__ import annotations

import json

ACTIONS = frozenset({"up", "down", "left", "right", "reset"})


def parse_action(text: str) -> str:
    """Validate the VLM's one-action JSON response."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Response must be a JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"action"}:
        raise ValueError('Response must contain exactly one "action" field')
    action = value["action"]
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError(f"Unsupported action {action!r}")
    return action
