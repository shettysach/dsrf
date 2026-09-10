"""Viewer-focused arrow-key controls for the directional Sokoban planner."""

from __future__ import annotations

import os
import socket
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from shared.arrow import agent_command_to_arrow, observation_from_arrow
from shared.messages import AgentCommand, VisualObservation

if TYPE_CHECKING:
    from dora import Node


_DIRECTIONS = {
    "up": "forward",
    "down": "backward",
    "left": "left",
    "right": "right",
}


class KeyboardSokobanAgentLoop:
    """Issue one planner command per observation from a viewer key press."""

    def __init__(
        self,
        node: Node,
        keyboard_socket: Path | None = None,
        *,
        read_key: Callable[[], str] | None = None,
    ) -> None:
        self.node = node
        self._keyboard_socket = keyboard_socket
        self._read_key = read_key
        self.observation: VisualObservation | None = None
        self.pending_command: str | None = None
        self.finished = False

    def run(self) -> None:
        if self._read_key is None:
            if self._keyboard_socket is None:
                raise ValueError("Keyboard Sokoban control requires KEYBOARD_SOCKET")
            with _ViewerKeyReceiver(self._keyboard_socket) as receiver:
                self._read_key = lambda: receiver.recv(16).decode("ascii")
                self._run_events()
        else:
            self._run_events()

    def _run_events(self) -> None:
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
            if (
                observation.observation_id != 0
                or observation.completed_command is not None
            ):
                raise RuntimeError("Initial observation must be observation 0")
        else:
            expected_id = self.observation.observation_id + 1
            if observation.observation_id != expected_id:
                raise RuntimeError(
                    f"Expected observation {expected_id}, got {observation.observation_id}"
                )
            if observation.completed_command != self.pending_command:
                raise RuntimeError(
                    "Completed command does not match the keyboard command"
                )

        self.observation = observation
        self.pending_command = None
        if not self.finished:
            self._send_key_command()

    def _send_key_command(self) -> None:
        assert self.observation is not None
        assert self._read_key is not None
        while True:
            key = self._read_key()
            if key == "finish":
                self._send(
                    AgentCommand(
                        self.observation.observation_id,
                        "finished",
                        "stand",
                        (),
                        terminal=True,
                    )
                )
                self.finished = True
                return
            direction = _DIRECTIONS.get(key)
            if direction is not None:
                self._send(
                    AgentCommand(
                        self.observation.observation_id,
                        f'{{"motion":"walk","direction":"{direction}"}}',
                        "walk",
                        (),
                        direction=direction,
                    )
                )
                return

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


class _ViewerKeyReceiver:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def __enter__(self) -> socket.socket:
        if self.path.exists():
            mode = self.path.stat().st_mode
            if not stat.S_ISSOCK(mode):
                raise RuntimeError(f"KEYBOARD_SOCKET is not a socket: {self.path}")
            self.path.unlink()
        self.socket.bind(str(self.path))
        return self.socket

    def __exit__(self, *_: object) -> None:
        self.socket.close()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
