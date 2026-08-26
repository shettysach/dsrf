from __future__ import annotations

from tasks.box_push import TASK as BOX_PUSH
from tasks.portrait_corridors import TASK as PORTRAIT_CORRIDORS
from tasks.seesaw import TASK as SEESAW
from tasks.sokoban import TASK as SOKOBAN
from tasks.spec import TaskSpec
from tasks.stairs import TASK as STAIRS

_TASK_SPECS = (BOX_PUSH, PORTRAIT_CORRIDORS, SEESAW, SOKOBAN, STAIRS)

TASKS: dict[str, TaskSpec] = {task.name: task for task in _TASK_SPECS}

if len(TASKS) != len(_TASK_SPECS):
    raise RuntimeError("Task names must be unique")


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task {name!r}. Available: {available}") from None
