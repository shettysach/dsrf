from __future__ import annotations

from script.interface import TaskScript
from script.tasks.push import PushScript


def create_task_script(task_name: str, prompt: str) -> TaskScript:
    """Return the script registered for a task name."""
    match task_name:
        case "box_push":
            return PushScript(motion_prompt=prompt)
        case _:
            raise ValueError(f"No script is registered for task {task_name!r}")
