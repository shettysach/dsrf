"""Task-specific command scripts.

Scripts are deliberately independent of the simulator and motion-generator
backends.  They only produce the same ``AgentCommand`` sent by a VLM agent.
"""

from script.interface import TaskScript
from script.registry import create_task_script

__all__ = ("TaskScript", "create_task_script")
