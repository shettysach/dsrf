import json
from typing import Any, cast

import numpy as np

from agent.vlm import OAIChatClient
from nodes.agent import AgentLoop
from shared.arrow import (
    agent_command_from_arrow,
    grounding_request_from_arrow,
    grounding_result_to_arrow,
    observation_to_arrow,
    pipeline_error_to_arrow,
)
from shared.messages import (
    EndEffectorTarget,
    GroundingResult,
    PipelineError,
    VisualObservation,
)


class _Response:
    def __init__(self, command: str) -> None:
        self.payload = json.dumps(
            {"choices": [{"message": {"content": command}}]}
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def read(self) -> bytes:
        return self.payload


def test_llama_client_uses_blank_model_and_replays_history(monkeypatch) -> None:
    posted: list[dict[str, Any]] = []
    responses = iter(["stand", "walk forward 0.4"])

    def urlopen(request, timeout):
        posted.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data),
            }
        )
        return _Response(next(responses))

    monkeypatch.setattr("agent.vlm.urllib.request.urlopen", urlopen)
    client = OAIChatClient(
        base_url="http://127.0.0.1:8080/",
        timeout=12.0,
        system_prompt="System file prompt.\n",
        user_prompt="User file prompt.\n",
    )
    first = VisualObservation(0, None, b"first")
    assert client.complete(first) == "stand"
    client.commit(first, "stand")

    second = VisualObservation(1, "stand", b"second")
    assert client.complete(second) == "walk forward 0.4"

    assert posted[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert posted[0]["timeout"] == 12.0
    assert posted[0]["payload"]["model"] == ""
    messages = posted[1]["payload"]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == "System file prompt.\n"
    assert messages[1]["content"][0]["text"].endswith("User file prompt.\n")
    assert messages[2]["content"] == "stand"
    assert messages[1]["content"][1]["image_url"]["url"].endswith("Zmlyc3Q=")
    assert messages[3]["content"][1]["image_url"]["url"].endswith("c2Vjb25k")


class _Node:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = iter(events)
        self.outputs: list[tuple[str, object, dict[str, object]]] = []
        self.logs: list[tuple[str, str, dict[str, object]]] = []

    def __iter__(self):
        return self.events

    def send_output(self, output_id, value, **kwargs) -> None:
        self.outputs.append((output_id, value, kwargs))

    def log(self, level, message, **kwargs) -> None:
        self.logs.append((level, message, kwargs))


class _Client:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.feedback: list[str | None] = []
        self.commits: list[tuple[int, str]] = []

    def complete(self, observation, *, retry_feedback=None) -> str:
        self.feedback.append(retry_feedback)
        return next(self.responses)

    def commit(self, observation, command) -> None:
        self.commits.append((observation.observation_id, command))


def _observation_event(observation: VisualObservation) -> dict[str, object]:
    value, metadata = observation_to_arrow(observation)
    return {
        "type": "INPUT",
        "id": "observation",
        "value": value,
        "metadata": metadata,
    }


def _grounding_event(
    observation_id: int,
    target_xys: tuple[tuple[float, float], ...] = ((1.0, 0.0),),
    end_effectors: tuple[EndEffectorTarget, ...] = (),
) -> dict[str, object]:
    value, metadata = grounding_result_to_arrow(
        GroundingResult(observation_id, target_xys, end_effectors)
    )
    return {
        "type": "INPUT",
        "id": "grounding_result",
        "value": value,
        "metadata": metadata,
    }


def _error_event(observation_id: int) -> dict[str, object]:
    return {
        "type": "INPUT",
        "id": "planning_error",
        "value": pipeline_error_to_arrow(
            PipelineError("motion-gen", observation_id, "bad command")
        ),
    }


def test_agent_retries_three_invalid_responses_then_stands() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            {"type": "STOP"},
        ]
    )
    client = _Client(["invalid one", "invalid two", "invalid three"])

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    commands = [
        agent_command_from_arrow(value, cast(Any, kwargs["metadata"]))
        for output_id, value, kwargs in node.outputs
        if output_id == "command"
    ]
    assert [command.text for command in commands] == [
        '{"motion":"stand"}',
    ]
    assert client.feedback[0] is None
    assert all(feedback is not None for feedback in client.feedback[1:])
    vlm_messages = [message for _, message, _ in node.logs if "VLM command" in message]
    assert len(vlm_messages) == 3
    assert "[OBS 0] VLM command: 'invalid one'" in vlm_messages[0]
    assert vlm_messages[0].endswith("retry=0")
    assert "[OBS 0] VLM command: 'invalid two'" in vlm_messages[1]
    assert vlm_messages[1].endswith("retry=1")
    assert "[OBS 0] VLM command: 'invalid three'" in vlm_messages[2]
    assert vlm_messages[2].endswith("retry=2")
    assert any(
        'fallback command: \'{"motion":"stand"}\'' in message
        for _, message, _ in node.logs
    )


def test_agent_commits_exact_completed_command() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"first")),
            _grounding_event(0),
            _observation_event(
                VisualObservation(
                    1,
                    '{"motion":"walk","waypoints_2d":[[500,500]]}',
                    b"second",
                )
            ),
            {"type": "STOP"},
        ]
    )
    client = _Client(
        [
            '{"motion":"walk","waypoints_2d":[[500,500]]}',
            '{"motion":"stand","waypoints_2d":[]}',
        ]
    )

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    assert client.commits == [(0, '{"motion":"walk","waypoints_2d":[[500,500]]}')]


def test_agent_command_without_waypoint_bypasses_grounding() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            {"type": "STOP"},
        ]
    )
    client = _Client(['{"motion":"stand"}'])

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    command = agent_command_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert command.motion == "stand"
    assert command.target_xys == ()
    assert [output_id for output_id, _, _ in node.outputs] == ["command"]


def test_agent_grounds_end_effector_before_sending_command() -> None:
    target = EndEffectorTarget("right_hand", (0.5, -0.1, 0.8))
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            _grounding_event(0, (), (target,)),
            {"type": "STOP"},
        ]
    )
    client = _Client(
        [
            '{"motion":"reach for the cup","end_effectors":'
            '[{"name":"right_hand","target_2d":[600,400]}]}'
        ]
    )

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    request_output = next(
        output for output in node.outputs if output[0] == "grounding_request"
    )
    request = grounding_request_from_arrow(
        request_output[1], cast(Any, request_output[2]["metadata"])
    )
    assert request.waypoints_2d == ()
    assert request.end_effectors_2d[0].name == "right_hand"
    assert request.end_effectors_2d[0].target_2d == (600, 400)
    command_output = next(output for output in node.outputs if output[0] == "command")
    command = agent_command_from_arrow(
        command_output[1], cast(Any, command_output[2]["metadata"])
    )
    assert command.end_effectors[0].name == target.name
    np.testing.assert_allclose(
        command.end_effectors[0].target_xyz, target.target_xyz
    )


def test_agent_retries_motion_gen_errors() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            _grounding_event(0),
            _error_event(0),
            _grounding_event(0),
            {"type": "STOP"},
        ]
    )
    client = _Client(
        [
            '{"motion":"walk","waypoints_2d":[[500,500]]}',
            '{"motion":"walk","waypoints_2d":[[500,500]]}',
        ]
    )

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    commands = [
        agent_command_from_arrow(value, cast(Any, kwargs["metadata"]))
        for output_id, value, kwargs in node.outputs
        if output_id == "command"
    ]
    assert [command.text for command in commands] == [
        '{"motion":"walk","waypoints_2d":[[500,500]]}',
        '{"motion":"walk","waypoints_2d":[[500,500]]}',
    ]
    assert client.feedback[1] is not None


def test_agent_requests_grounding_before_sending_complete_command() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            _grounding_event(0, ((0.8, -0.2), (0.2, 0.6))),
            {"type": "STOP"},
        ]
    )
    client = _Client(
        ['{"motion":"sidestep carefully","waypoints_2d":[[300,600],[600,500]]}']
    )

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    request_output = next(
        output for output in node.outputs if output[0] == "grounding_request"
    )
    request = grounding_request_from_arrow(
        request_output[1], cast(Any, request_output[2]["metadata"])
    )
    assert request.waypoints_2d == ((300, 600), (600, 500))
    command_output = next(output for output in node.outputs if output[0] == "command")
    command = agent_command_from_arrow(
        command_output[1], cast(Any, command_output[2]["metadata"])
    )
    assert command.motion == "sidestep carefully"
    np.testing.assert_allclose(command.target_xys, ((0.8, -0.2), (0.2, 0.6)))
