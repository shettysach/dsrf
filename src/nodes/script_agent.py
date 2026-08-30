"""Dora agent loop for task scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.arrow import agent_command_to_arrow, observation_from_arrow
from shared.messages import VisualObservation

if TYPE_CHECKING:
    from dora import Node

    from script.interface import TaskScript


class ScriptAgentLoop:
    """Feed task-script commands into the normal AgentCommand pipeline."""

    def __init__(self, node: Node, script: TaskScript) -> None:
        self.node = node
        self.script = script
        self.observation: VisualObservation | None = None
        self.pending_command: str | None = None

    def run(self) -> None:
        for event in self.node:
            if event["type"] == "STOP":
                return
            if event["type"] != "INPUT":
                continue
            if event["id"] == "observation":
                self._accept_observation(
                    observation_from_arrow(
                        event["value"], dict(event.get("metadata") or {})
                    )
                )
            elif event["id"] in {"planning_error", "sim_error"}:
                raise RuntimeError("A scripted command failed in the pipeline")

    def _accept_observation(self, observation: VisualObservation) -> None:
        if self.observation is None:
            if observation.observation_id != 0:
                raise RuntimeError(
                    f"First observation must be 0, got {observation.observation_id}"
                )
            if observation.completed_command is not None:
                raise RuntimeError("Initial observation has a completed command")
        else:
            expected_id = self.observation.observation_id + 1
            if observation.observation_id != expected_id:
                raise RuntimeError(
                    f"Expected observation {expected_id}, got {observation.observation_id}"
                )
            if observation.completed_command != self.pending_command:
                raise RuntimeError(
                    "Completed command does not match the scripted command sent for "
                    "the previous observation"
                )

        self.observation = observation
        self.pending_command = None
        command = self.script.next_command(observation.observation_id)
        if command is None:
            return
        if command.observation_id != observation.observation_id:
            raise RuntimeError(
                f"Script emitted command for observation {command.observation_id}, "
                f"expected {observation.observation_id}"
            )
        data, metadata = agent_command_to_arrow(command)
        self.node.send_output("command", data, metadata=metadata)
        self.pending_command = command.text
        self.node.log(
            "info",
            f"[OBS {command.observation_id}] script command: {command.text!r}",
            target="dsrf.agent.script",
            fields={
                "event": "script_command",
                "observation_id": str(command.observation_id),
                "command": command.text,
            },
        )
