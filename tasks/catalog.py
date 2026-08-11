from __future__ import annotations

from tasks.portrait_corridors import TASK as PORTRAIT_CORRIDORS
from tasks.spec import TaskSpec

_TASK_SPECS = (PORTRAIT_CORRIDORS,)

TASKS: dict[str, TaskSpec] = {task.name: task for task in _TASK_SPECS}

if len(TASKS) != len(_TASK_SPECS):
    raise RuntimeError("Task names must be unique")


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task {name!r}. Available: {available}") from None
