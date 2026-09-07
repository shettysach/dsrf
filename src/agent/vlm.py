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
    reasoning: str | None = None
    execution_feedback: str | None = None


class OAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
        recent_turns: int = 3,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("System prompt must not be empty")
        if not user_prompt.strip():
            raise ValueError("User prompt must not be empty")
        if recent_turns < 0:
            raise ValueError("recent_turns must be non-negative")
        self.endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.tool = tool
        self.tool_name = _tool_name(tool)
        self.recent_turns = recent_turns
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
        # The current observation completely describes Sokoban state. Replay a
        # bounded suffix of old turns as text/tool context only; sending an old
        # board image makes moved boxes and the robot ambiguous.
        history = self._history[-self.recent_turns :] if self.recent_turns else ()
        for turn in history:
            _append_turn(messages, turn, user_prompt=self.user_prompt)
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


def _append_turn(
    messages: list[dict[str, Any]],
    turn: _ConversationTurn,
    *,
    user_prompt: str,
) -> None:
    messages.append(_history_user_message(turn.observation, user_prompt))
    messages.append(turn.completion.assistant_message)
    if turn.completion.tool_call_id is not None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": turn.completion.tool_call_id,
                "content": turn.completion.execution_feedback or "Motion completed.",
            }
        )


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
    reasoning = message.get(
        "reasoning_content", message.get("reasoning", message.get("content"))
    )
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return CommandCompletion(
        function["arguments"], assistant_message, tool_call_id, reasoning
    )


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
    if observation.execution_feedback is not None:
        text = f"Completed command: {completed}\nSimulator events: {observation.execution_feedback}\n\n{user_prompt}"
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


def _history_user_message(
    observation: VisualObservation, user_prompt: str
) -> dict[str, Any]:
    """Preserve an old turn's command context without its stale image."""
    completed = observation.completed_command or "none (initial observation)"
    return {
        "role": "user",
        "content": f"Completed command: {completed}\n\n{user_prompt}",
    }
