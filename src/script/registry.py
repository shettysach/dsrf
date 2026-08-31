from __future__ import annotations

from script.interface import TaskScript
from script.tasks.prompt import PromptScript
from script.tasks.push import PushScript
from script.tasks.right_kick import RightKickScript


def create_task_script(task_name: str, prompt: str) -> TaskScript:
    """Return the script registered for a task name."""
    match task_name:
        case "box_push":
            return PushScript(prompt=prompt)
        case "prompt":
            return PromptScript(prompt=prompt)
        case "right_kick":
            return RightKickScript(prompt=prompt)
        case _:
            raise ValueError(f"No script is registered for task {task_name!r}")
