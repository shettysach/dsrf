from __future__ import annotations

import json
import os
import select
import subprocess
import threading
import time
import uuid
from base64 import b64encode
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.pi_debug import PiDebug
from shared.messages import VisualObservation


@dataclass(frozen=True)
class PiAction:
    motion: str
    direction: str | None
    waypoints_2d: tuple[tuple[int, int], ...]

    @property
    def text(self) -> str:
        payload: dict[str, object] = {"motion": self.motion}
        if self.direction is not None:
            payload["direction"] = self.direction
        else:
            payload["waypoints_2d"] = [list(point) for point in self.waypoints_2d]
        return json.dumps(payload, separators=(",", ":"))


class PiRpcClient:
    """A single persistent Pi RPC session for one closed-loop agent run."""

    def __init__(
        self,
        *,
        timeout: float,
        system_prompt: str,
        command_mode: str,
        provider: str | None = None,
        model: str | None = None,
        debug: PiDebug | None = None,
        command: Sequence[str] = ("pi",),
    ) -> None:
        self.timeout = timeout
        self._debug = debug or PiDebug()
        self._stderr: deque[str] = deque(maxlen=40)
        # Pi owns provider discovery, credentials, endpoint, and model choice.
        # In particular, pi-llama-cpp may register a provider name such as
        # ``llama-server=http://127.0.0.1:8080`` in settings.json.  Do not
        # replace that with Pi's built-in ``llama.cpp`` provider: leaving the
        # flag out lets Pi honor defaultProvider from the user's config.
        selected_provider = provider or os.environ.get("PI_PROVIDER")
        selected_model = model or os.environ.get("PI_MODEL")
        pi_command = [*command, "--mode", "rpc"]
        if selected_provider:
            pi_command.extend(("--provider", selected_provider))
        if selected_model:
            pi_command.extend(("--model", selected_model))
        pi_command.extend(
            (
                "--no-builtin-tools",
                "--extension",
                str(_extension_path()),
                "--extension",
                str(_context_extension_path()),
                "--no-context-files",
                "--no-skills",
                "--no-prompt-templates",
                "--system-prompt",
                system_prompt,
            )
        )
        self._process = subprocess.Popen(
            pi_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env={**os.environ, "DSRF_COMMAND_MODE": command_mode},
        )
        self._debug.started(command_mode=command_mode, provider=selected_provider)
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._collect_stderr, name="dsrf-pi-stderr", daemon=True
        )
        self._stderr_thread.start()

    def complete(
        self,
        observation: VisualObservation,
        *,
        retry_feedback: str | None = None,
    ) -> PiAction:
        self._debug.prompt(
            observation_id=observation.observation_id,
            completed_command=observation.completed_command,
            retry=retry_feedback is not None,
        )
        message = _observation_text(observation)
        if retry_feedback is not None:
            message += (
                f"\n\nYour previous response was invalid: {retry_feedback}\n"
                "Call robot_action with a corrected action."
            )
        request: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "type": "prompt",
            "message": message,
            "images": _current_images(observation),
        }
        self._send(request)
        return self._read_response(str(request["id"]))

    def close(self) -> None:
        if self._process.poll() is not None:
            self._debug.stopped()
            return
        assert self._process.stdin is not None
        self._process.stdin.close()
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._debug.stopped()

    def reset(self) -> None:
        """Start a fresh Pi conversation when the simulator starts a new run."""
        request = {"id": str(uuid.uuid4()), "type": "new_session"}
        self._send(request)
        self._read_control_response(str(request["id"]))

    def _send(self, request: dict[str, Any]) -> None:
        if self._process.poll() is not None:
            raise RuntimeError(
                self._exit_detail("Pi exited before receiving a request")
            )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(
                json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._exit_detail("Pi RPC stdin closed")) from exc

    def _read_response(self, request_id: str) -> PiAction:
        assert self._process.stdout is not None
        deadline = time.monotonic() + self.timeout
        action: PiAction | None = None
        accepted = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Pi did not settle within {self.timeout:.1f}s")
            readable, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not readable:
                continue
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(self._exit_detail("Pi RPC stdout closed"))
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Pi emitted invalid JSONL: {line!r}") from exc
            self._debug.event(event)
            if event.get("type") == "response" and event.get("id") == request_id:
                if not event.get("success"):
                    raise RuntimeError(
                        f"Pi rejected prompt: {event.get('error', event)}"
                    )
                accepted = True
            elif event.get("type") == "extension_error":
                raise RuntimeError(f"Pi extension error: {event.get('error', event)}")
            elif event.get("type") == "tool_execution_start":
                if event.get("toolName") == "robot_action":
                    if action is not None:
                        self._debug.duplicate_robot_action()
                    else:
                        action = _action_from_arguments(event.get("args"))
            elif event.get("type") == "agent_settled":
                if not accepted:
                    raise RuntimeError("Pi settled before accepting the prompt")
                if action is None:
                    raise RuntimeError("Pi settled without invoking robot_action")
                return action

    def _read_control_response(self, request_id: str) -> None:
        assert self._process.stdout is not None
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Pi did not respond within {self.timeout:.1f}s")
            readable, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not readable:
                continue
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(self._exit_detail("Pi RPC stdout closed"))
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Pi emitted invalid JSONL: {line!r}") from exc
            self._debug.event(event)
            if event.get("type") == "response" and event.get("id") == request_id:
                if not event.get("success"):
                    raise RuntimeError(
                        f"Pi rejected session reset: {event.get('error', event)}"
                    )
                return
            if event.get("type") == "extension_error":
                raise RuntimeError(f"Pi extension error: {event.get('error', event)}")

    def _collect_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in iter(self._process.stderr.readline, b""):
            detail = line.decode("utf-8", errors="replace").rstrip()
            self._stderr.append(detail)
            self._debug.stderr(detail)

    def _exit_detail(self, prefix: str) -> str:
        stderr = "\n".join(self._stderr)
        suffix = f" (exit code {self._process.poll()})"
        return f"{prefix}{suffix}" if not stderr else f"{prefix}{suffix}: {stderr}"


def _observation_text(observation: VisualObservation) -> str:
    completed = observation.completed_command or "none (initial observation)"
    collision = (
        "\nCollision happened during the completed command. Reassess carefully."
        if observation.collision_detected
        else ""
    )
    return (
        f"Observation {observation.observation_id}.\n"
        f"Completed command: {completed}{collision}\n\n"
        "The first image is the current robot-camera RGB observation."
        + (
            " The second image is only the robot's top-down path so far; it has no "
            "map, objects, goals, or world-coordinate labels."
            if observation.trajectory_png is not None
            else ""
        )
        + "\n\nChoose the robot's next command from this current observation."
    )


def _current_images(observation: VisualObservation) -> list[dict[str, str]]:
    images = [
        {
            "type": "image",
            "data": b64encode(observation.jpeg).decode("ascii"),
            "mimeType": "image/jpeg",
        }
    ]
    if observation.trajectory_png is not None:
        images.append(
            {
                "type": "image",
                "data": b64encode(observation.trajectory_png).decode("ascii"),
                "mimeType": "image/png",
            }
        )
    return images


def _action_from_arguments(arguments: object) -> PiAction:
    if not isinstance(arguments, dict):
        raise ValueError("robot_action arguments must be an object")
    motion = arguments.get("motion")
    if not isinstance(motion, str) or not motion.strip():
        raise ValueError("robot_action motion must be non-empty text")
    direction = arguments.get("direction")
    waypoints = arguments.get("waypoints_2d")
    if direction is not None and waypoints is not None:
        raise ValueError("robot_action cannot contain both direction and waypoints_2d")
    if direction is not None:
        if not isinstance(direction, str):
            raise ValueError("robot_action direction must be text")
        return PiAction(motion.strip(), direction, ())
    if not isinstance(waypoints, list):
        raise ValueError("robot_action requires direction or waypoints_2d")
    normalized: list[tuple[int, int]] = []
    for waypoint in waypoints:
        if (
            not isinstance(waypoint, list)
            or len(waypoint) != 2
            or any(type(value) is not int for value in waypoint)
            or not all(0 <= value <= 1000 for value in waypoint)
        ):
            raise ValueError("robot_action waypoints must be [x, y] integers in [0,1000]")
        normalized.append((waypoint[0], waypoint[1]))
    return PiAction(motion.strip(), None, tuple(normalized))


def _extension_path() -> Path:
    return Path(__file__).resolve().parents[2] / "pi" / "kinematic_planner.ts"


def _context_extension_path() -> Path:
    return Path(__file__).resolve().parents[2] / "pi" / "dsrf_context.ts"
