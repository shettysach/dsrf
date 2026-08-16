from __future__ import annotations

from agent.debug import AgentDebug


class _Node:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def log(self, level: str, message: str, **kwargs: object) -> None:
        self.calls.append((level, message, kwargs))


def test_agent_debug_prints_reasoning_and_pretty_json() -> None:
    node = _Node()

    AgentDebug(True).response(
        node,
        observation_id=3,
        reasoning="The box is aligned with the goal.",
        command='{"direction":"left","motion":"walk"}',
    )

    assert node.calls == [
        (
            "warn",
            "[OBS 3] VLM reasoning:\nThe box is aligned with the goal.\n"
            "[OBS 3] VLM command:\n{\n  \"direction\": \"left\",\n"
            '  "motion": "walk"\n}',
            {
                "target": "dsrf.agent.debug",
                "fields": {
                    "event": "agent_debug_response",
                    "observation_id": "3",
                },
            },
        )
    ]


def test_agent_debug_is_silent_when_disabled() -> None:
    node = _Node()

    AgentDebug().response(
        node, observation_id=0, reasoning=None, command='{"motion":"stand"}'
    )

    assert node.calls == []
