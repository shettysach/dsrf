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
from typing import Any

from shared.messages import VisualObservation


class PiRpcClient:
    """A single persistent Pi RPC session for one closed-loop agent run."""

    def __init__(
        self,
        *,
        timeout: float,
        system_prompt: str,
        command: Sequence[str] = ("pi",),
    ) -> None:
        self.timeout = timeout
        self._stderr: deque[str] = deque(maxlen=40)
        self._process = subprocess.Popen(
            [
                *command,
                "--mode",
                "rpc",
                "--no-tools",
                "--no-context-files",
                "--append-system-prompt",
                system_prompt,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=os.environ.copy(),
        )
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
    ) -> str:
        if retry_feedback is None:
            request: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "type": "prompt",
                "message": _observation_text(observation),
                "images": [
                    {
                        "type": "image",
                        "data": b64encode(observation.jpeg).decode("ascii"),
                        "mimeType": "image/jpeg",
                    }
                ],
            }
        else:
            request = {
                "id": str(uuid.uuid4()),
                "type": "prompt",
                "message": (
                    f"Your previous response was invalid: {retry_feedback}\n\n"
                    "Return a corrected command for the current observation."
                ),
            }
        self._send(request)
        return self._read_response(str(request["id"]))

    def close(self) -> None:
        if self._process.poll() is not None:
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

    def _send(self, request: dict[str, Any]) -> None:
        if self._process.poll() is not None:
            raise RuntimeError(self._exit_detail("Pi exited before receiving a request"))
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(
                json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._exit_detail("Pi RPC stdin closed")) from exc

    def _read_response(self, request_id: str) -> str:
        assert self._process.stdout is not None
        deadline = time.monotonic() + self.timeout
        final_text: str | None = None
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
            if event.get("type") == "response" and event.get("id") == request_id:
                if not event.get("success"):
                    raise RuntimeError(f"Pi rejected prompt: {event.get('error', event)}")
                accepted = True
            elif event.get("type") == "extension_error":
                raise RuntimeError(f"Pi extension error: {event.get('error', event)}")
            elif event.get("type") == "message_end":
                message = event.get("message")
                if isinstance(message, dict) and message.get("role") == "assistant":
                    text = _assistant_text(message)
                    if text:
                        final_text = text
            elif event.get("type") == "agent_settled":
                if not accepted:
                    raise RuntimeError("Pi settled before accepting the prompt")
                if final_text is None:
                    raise RuntimeError("Pi settled without an assistant text response")
                return final_text

    def _collect_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in iter(self._process.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", errors="replace").rstrip())

    def _exit_detail(self, prefix: str) -> str:
        stderr = "\n".join(self._stderr)
        suffix = f" (exit code {self._process.poll()})"
        return f"{prefix}{suffix}" if not stderr else f"{prefix}{suffix}: {stderr}"


def _observation_text(observation: VisualObservation) -> str:
    completed = observation.completed_command or "none (initial observation)"
    return (
        f"Observation {observation.observation_id}.\n"
        f"Completed command: {completed}\n\n"
        "Choose the robot's next command from this current observation."
    )


def _assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ).strip()
