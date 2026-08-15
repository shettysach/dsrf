"""Human-readable, opt-in diagnostics for the Pi RPC child process."""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Mapping
from datetime import datetime
from typing import Any, TextIO


class PiDebug:
    """Format the useful Pi RPC lifecycle events without exposing image bytes."""

    def __init__(self, enabled: bool | None = None, *, stream: TextIO | None = None) -> None:
        self.enabled = _enabled_from_env() if enabled is None else enabled
        self._stream = sys.stderr if stream is None else stream
        self._lock = threading.Lock()
        self._text_open = False

    def started(self, *, command_mode: str, provider: str | None) -> None:
        selection = provider or "Pi default"
        self._write(f"started (provider={selection}, command_mode={command_mode})")

    def prompt(self, *, observation_id: int, completed_command: str | None, retry: bool) -> None:
        if retry:
            self._write(f"retrying observation {observation_id} without image")
            return
        completed = completed_command or "initial observation"
        self._write(f"prompting observation {observation_id} (image attached; completed={completed})")

    def event(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "message_update":
            self._message_update(event.get("assistantMessageEvent"))
        elif event_type == "tool_execution_start":
            self._finish_text()
            self._write(
                f"tool {event.get('toolName', '<unknown>')} "
                f"{_compact(event.get('args', {}))}"
            )
        elif event_type == "tool_execution_end":
            outcome = "failed" if event.get("isError") else "completed"
            self._write(f"tool {event.get('toolName', '<unknown>')} {outcome}")
        elif event_type == "agent_settled":
            self._finish_text()
            self._write("settled")
        elif event_type == "extension_error":
            self._finish_text()
            self._write(f"extension error: {event.get('error', event)}")
        elif event_type == "auto_retry_start":
            self._write(
                f"provider retry {event.get('attempt')}/{event.get('maxAttempts')}: "
                f"{event.get('errorMessage', 'unknown error')}"
            )
        elif event_type == "compaction_start":
            self._write(f"compacting context ({event.get('reason', 'unknown reason')})")
        elif event_type == "compaction_end":
            self._write("context compaction finished")

    def stderr(self, line: str) -> None:
        if line:
            self._finish_text()
            self._write(f"stderr: {line}")

    def error(self, detail: str) -> None:
        self._finish_text()
        self._write(f"error: {detail}")

    def stopped(self) -> None:
        self._finish_text()
        self._write("stopped")

    def _message_update(self, assistant_event: object) -> None:
        if not isinstance(assistant_event, Mapping):
            return
        if assistant_event.get("type") == "text_start":
            self._finish_text()
            self._write_prefix("assistant: ")
            self._text_open = True
        elif assistant_event.get("type") == "text_delta":
            if not self._text_open:
                self._write_prefix("assistant: ")
                self._text_open = True
            self._write_raw(str(assistant_event.get("delta", "")))
        elif assistant_event.get("type") == "text_end":
            self._finish_text()

    def _finish_text(self) -> None:
        if self.enabled and self._text_open:
            with self._lock:
                print(file=self._stream, flush=True)
            self._text_open = False

    def _write(self, message: str) -> None:
        if self.enabled:
            with self._lock:
                print(f"[pi {datetime.now().strftime('%H:%M:%S')}] {message}", file=self._stream, flush=True)

    def _write_prefix(self, prefix: str) -> None:
        if self.enabled:
            with self._lock:
                print(f"[pi {datetime.now().strftime('%H:%M:%S')}] {prefix}", end="", file=self._stream, flush=True)

    def _write_raw(self, text: str) -> None:
        if self.enabled:
            with self._lock:
                print(text, end="", file=self._stream, flush=True)


def _enabled_from_env() -> bool:
    return os.environ.get("PI_DEBUG", "").strip().lower() not in {"", "0", "false", "no", "off"}


def _compact(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)
