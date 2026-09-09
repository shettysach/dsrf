"""Terminal controls for the directional Sokoban planner."""

from __future__ import annotations

import sys
import termios
import tty
from collections.abc import Callable
from typing import TYPE_CHECKING

from shared.arrow import agent_command_to_arrow, observation_from_arrow
from shared.messages import AgentCommand, VisualObservation

if TYPE_CHECKING:
    from dora import Node


_DIRECTIONS = {
    "w": "forward",
    "a": "left",
    "s": "backward",
    "d": "right",
    "h": "left",
    "j": "backward",
    "k": "forward",
    "l": "right",
}


class KeyboardSokobanAgentLoop:
    """Issue one planner command per observation from a terminal key press."""

    def __init__(
        self, node: Node, *, read_key: Callable[[], str] | None = None
    ) -> None:
        self.node = node
        self._read_key = read_key or _read_key
        self.observation: VisualObservation | None = None
        self.pending_command: str | None = None
        self.finished = False

    def run(self) -> None:
        for event in self.node:
            if event["type"] == "STOP":
                return
            if event["type"] != "INPUT" or event["id"] != "observation":
                continue
            self._accept_observation(
                observation_from_arrow(
                    event["value"], dict(event.get("metadata") or {})
                )
            )

    def _accept_observation(self, observation: VisualObservation) -> None:
        if self.observation is None:
            if observation.observation_id != 0 or observation.completed_command is not None:
                raise RuntimeError("Initial observation must be observation 0")
        else:
            expected_id = self.observation.observation_id + 1
            if observation.observation_id != expected_id:
                raise RuntimeError(
                    f"Expected observation {expected_id}, got {observation.observation_id}"
                )
            if observation.completed_command != self.pending_command:
                raise RuntimeError("Completed command does not match the keyboard command")

        self.observation = observation
        self.pending_command = None
        if self.finished:
            return
        self._send_key_command()

    def _send_key_command(self) -> None:
        assert self.observation is not None
        while True:
            print("Sokoban: WASD/HJKL move; F finishes.", flush=True)
            key = self._read_key().lower()
            if key == "f":
                command = AgentCommand(
                    self.observation.observation_id,
                    "finished",
                    "stand",
                    (),
                    terminal=True,
                )
                self._send(command)
                self.finished = True
                return
            direction = _DIRECTIONS.get(key)
            if direction is not None:
                command = AgentCommand(
                    self.observation.observation_id,
                    f'{{"motion":"walk","direction":"{direction}"}}',
                    "walk",
                    (),
                    direction=direction,
                )
                self._send(command)
                return
            print(f"Ignored key {key!r}; use WASD, HJKL, or F.", flush=True)

    def _send(self, command: AgentCommand) -> None:
        data, metadata = agent_command_to_arrow(command)
        self.node.send_output("command", data, metadata=metadata)
        self.pending_command = command.text
        self.node.log(
            "info",
            f"[OBS {command.observation_id}] keyboard command: {command.text!r}",
            target="dsrf.agent.keyboard",
            fields={
                "event": "keyboard_command",
                "observation_id": str(command.observation_id),
                "command": command.text,
            },
        )


def _read_key() -> str:
    if not sys.stdin.isatty():
        raise RuntimeError("Keyboard Sokoban control requires an interactive terminal")
    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
