"""Task scenes available to the DSRF runtime."""

from tasks.catalog import TASKS, get_task
from tasks.spec import TaskSpec, ViewerSpec

__all__ = ["TASKS", "TaskSpec", "ViewerSpec", "get_task"]
