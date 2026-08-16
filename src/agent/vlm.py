from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from dataclasses import dataclass
from typing import Any

from shared.messages import VisualObservation


@dataclass(frozen=True)
class _ConversationTurn:
    observation: VisualObservation
    completion: CommandCompletion


@dataclass(frozen=True)
class CommandCompletion:
    command: str
    assistant_message: dict[str, Any]
    tool_call_id: str | None


class OAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("System prompt must not be empty")
        if not user_prompt.strip():
            raise ValueError("User prompt must not be empty")
        self.endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.tool = tool
        self.tool_name = _tool_name(tool)
        self._history: list[_ConversationTurn] = []

    def complete(
        self,
        observation: VisualObservation,
        *,
        retry_feedback: str | None = None,
    ) -> CommandCompletion:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for turn in self._history:
            messages.append(_user_message(turn.observation, self.user_prompt))
            messages.append(turn.completion.assistant_message)
            if turn.completion.tool_call_id is not None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": turn.completion.tool_call_id,
                        "content": "Motion completed.",
                    }
                )
        messages.append(
            _user_message(
                observation,
                self.user_prompt,
                retry_feedback=retry_feedback,
            )
        )

        payload: dict[str, Any] = {
            "model": "",
            "messages": messages,
            "temperature": 0,
        }
        payload.update(
            tools=[self.tool],
            tool_choice={
                "type": "function",
                "function": {"name": self.tool_name},
            },
            parallel_tool_calls=False,
        )

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            document = json.loads(response.read().decode("utf-8"))

        try:
            message = document["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama-server returned no assistant message") from exc
        if not isinstance(message, dict):
            raise RuntimeError("llama-server returned an invalid assistant message")
        return _tool_completion(message, self.tool_name)

    def commit(
        self,
        observation: VisualObservation,
        completion: CommandCompletion,
    ) -> None:
        self._history.append(_ConversationTurn(observation, completion))


def _tool_completion(message: dict[str, Any], tool_name: str) -> CommandCompletion:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise RuntimeError("llama-server returned no motion tool call")
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise RuntimeError("llama-server returned an invalid motion tool call")
    tool_call_id = tool_call.get("id")
    if not isinstance(tool_call_id, str):
        raise RuntimeError("llama-server returned a motion tool call without an ID")
    function = tool_call.get("function")
    if (
        not isinstance(function, dict)
        or function.get("name") != tool_name
        or not isinstance(function.get("arguments"), str)
    ):
        raise RuntimeError("llama-server returned an invalid motion tool call")

    assistant_message = {"role": "assistant", "tool_calls": tool_calls}
    if isinstance(message.get("content"), str):
        assistant_message["content"] = message["content"]
    return CommandCompletion(function["arguments"], assistant_message, tool_call_id)


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("Tool must declare a function name")
    return function["name"]


def _user_message(
    observation: VisualObservation,
    user_prompt: str,
    *,
    retry_feedback: str | None = None,
) -> dict[str, Any]:
    completed = observation.completed_command or "none (initial observation)"
    text = f"Completed command: {completed}\n\n{user_prompt}"
    if retry_feedback is not None:
        text = f"{retry_feedback}\n\n{text}"
    image_url = "data:image/jpeg;base64," + b64encode(observation.jpeg).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }
