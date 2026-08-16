from __future__ import annotations

import json

ACTIONS = frozenset({"up", "down", "left", "right", "reset"})


def parse_action(text: str) -> str:
    """Normalize one optional Markdown fence, then validate the action JSON."""
    try:
        value = json.loads(_unwrap_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Response must be a JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"action"}:
        raise ValueError('Response must contain exactly one "action" field')
    action = value["action"]
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError(f"Unsupported action {action!r}")
    return action


def _unwrap_json_fence(text: str) -> str:
    """Accept exactly one outer ```json (or ```) block, with no surrounding prose."""
    stripped = text.strip()
    lines = stripped.splitlines()
    if not lines or not lines[0].strip().startswith("```"):
        return stripped
    opening = lines[0].strip().lower()
    if (
        opening not in {"```", "```json"}
        or len(lines) < 3
        or lines[-1].strip() != "```"
    ):
        return stripped
    return "\n".join(lines[1:-1]).strip()
