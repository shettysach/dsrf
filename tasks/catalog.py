from __future__ import annotations

from tasks.portrait_corridors import TASK as PORTRAIT_CORRIDORS
from tasks.sokoban import TASK as SOKOBAN
from tasks.spec import TaskSpec
from tasks.stack_steps import TASK as STACK_STEPS

_TASK_SPECS = (PORTRAIT_CORRIDORS, SOKOBAN, STACK_STEPS)

TASKS: dict[str, TaskSpec] = {task.name: task for task in _TASK_SPECS}

if len(TASKS) != len(_TASK_SPECS):
    raise RuntimeError("Task names must be unique")


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task {name!r}. Available: {available}") from None
