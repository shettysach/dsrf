"""Task scenes available to the DSRF runtime."""

from tasks.catalog import TASKS, get_task
from tasks.spec import ObservationCameraSpec, TaskSpec

__all__ = ["TASKS", "ObservationCameraSpec", "TaskSpec", "get_task"]
