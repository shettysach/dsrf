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
    assistant: str


class OAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("System prompt must not be empty")
        if not user_prompt.strip():
            raise ValueError("User prompt must not be empty")
        self.endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self._history: list[_ConversationTurn] = []

    def complete(
        self,
        observation: VisualObservation,
        *,
        retry_feedback: str | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for turn in self._history:
            messages.append(_user_message(turn.observation, self.user_prompt))
            messages.append({"role": "assistant", "content": turn.assistant})
        messages.append(
            _user_message(
                observation,
                self.user_prompt,
                retry_feedback=retry_feedback,
            )
        )

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {"model": "", "messages": messages, "temperature": 0},
                separators=(",", ":"),
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            document = json.loads(response.read().decode("utf-8"))

        try:
            content = document["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama-server returned no assistant message") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("llama-server returned an empty assistant message")
        return content.strip()

    def commit(self, observation: VisualObservation, command: str) -> None:
        # Projection depth is only needed for the current waypoint and must not
        # accumulate in the VLM conversation history.
        history_observation = VisualObservation(
            observation.observation_id,
            observation.completed_command,
            observation.jpeg,
        )
        self._history.append(_ConversationTurn(history_observation, command))


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
