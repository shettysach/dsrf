import json
import sys
from typing import Any, cast

import numpy as np

from agent.pi import PiAction, PiRpcClient
from nodes.agent import AgentLoop
from shared.arrow import (
    agent_command_from_arrow,
    grounding_request_from_arrow,
    grounding_result_to_arrow,
    observation_to_arrow,
    pipeline_error_to_arrow,
)
from shared.messages import GroundingResult, PipelineError, VisualObservation


def test_pi_rpc_client_sends_current_images_on_every_request(tmp_path) -> None:
    received = tmp_path / "received.jsonl"
    fake_pi = tmp_path / "fake_pi.py"
    fake_pi.write_text(
        "import json, pathlib, sys\n"
        "output = pathlib.Path(sys.argv[1])\n"
        "with output.open('w') as saved:\n"
        "  for index, line in enumerate(sys.stdin):\n"
        "    request = json.loads(line)\n"
        "    saved.write(json.dumps(request) + '\\n'); saved.flush()\n"
        "    print(json.dumps({'type':'response','id':request['id'],'command':'prompt','success':True}), flush=True)\n"
        "    print(json.dumps({'type':'tool_execution_start','toolName':'robot_action','args':{'motion':'walk','direction':'forward'}}), flush=True)\n"
        "    print(json.dumps({'type':'agent_settled'}), flush=True)\n"
    )
    client = PiRpcClient(
        timeout=2.0,
        system_prompt="Task prompt",
        command_mode="direction",
        command=(sys.executable, str(fake_pi), str(received)),
    )
    try:
        observation = VisualObservation(0, None, b"jpeg", trajectory_png=b"png")
        assert client.complete(observation).text == (
            '{"motion":"walk","direction":"forward"}'
        )
        assert (
            client.complete(observation, retry_feedback="bad JSON")
            == PiAction("walk", "forward", ())
        )
    finally:
        client.close()

    first, retry = [json.loads(line) for line in received.read_text().splitlines()]
    assert first["images"] == [
        {"type": "image", "data": "anBlZw==", "mimeType": "image/jpeg"},
        {"type": "image", "data": "cG5n", "mimeType": "image/png"},
    ]
    assert "Completed command: none (initial observation)" in first["message"]
    assert retry["images"] == first["images"]
    assert "bad JSON" in retry["message"]


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

    def complete(self, observation, *, retry_feedback=None) -> PiAction:
        self.feedback.append(retry_feedback)
        return _action(next(self.responses))


def _observation_event(observation: VisualObservation) -> dict[str, object]:
    value, metadata = observation_to_arrow(observation)
    return {
        "type": "INPUT",
        "id": "observation",
        "value": value,
        "metadata": metadata,
    }


def _action(text: str) -> PiAction:
    payload = json.loads(text)
    return PiAction(
        payload["motion"],
        payload.get("direction"),
        tuple(tuple(point) for point in payload.get("waypoints_2d", [])),
    )


def _grounding_event(
    observation_id: int, target_xys: tuple[tuple[float, float], ...] = ((1.0, 0.0),)
) -> dict[str, object]:
    value, metadata = grounding_result_to_arrow(
        GroundingResult(observation_id, target_xys)
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


def test_agent_retries_three_downstream_errors_then_stands() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            _grounding_event(0),
            _error_event(0),
            _grounding_event(0),
            _error_event(0),
            _grounding_event(0),
            _error_event(0),
            {"type": "STOP"},
        ]
    )
    client = _Client(
        [
            '{"motion":"walk","waypoints_2d":[[500,500]]}',
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
        '{"motion":"walk","waypoints_2d":[[500,500]]}',
        '{"motion":"stand","waypoints_2d":[]}',
    ]
    assert client.feedback[0] is None
    assert all(feedback is not None for feedback in client.feedback[1:])
    pi_messages = [message for _, message, _ in node.logs if "Pi action" in message]
    assert len(pi_messages) == 3
    assert "[OBS 0] Pi action: '{\"motion\":\"walk\",\"waypoints_2d\":[[500,500]]}'" in pi_messages[0]
    assert pi_messages[0].endswith("retry=0")
    assert "[OBS 0] Pi action: '{\"motion\":\"walk\",\"waypoints_2d\":[[500,500]]}'" in pi_messages[1]
    assert pi_messages[1].endswith("retry=1")
    assert "[OBS 0] Pi action: '{\"motion\":\"walk\",\"waypoints_2d\":[[500,500]]}'" in pi_messages[2]
    assert pi_messages[2].endswith("retry=2")
    assert any(
        'fallback command: \'{"motion":"stand","waypoints_2d":[]}\'' in message
        for _, message, _ in node.logs
    )


def test_agent_accepts_exact_completed_command_without_client_commit() -> None:
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

    assert client.feedback == [None, None]


def test_agent_command_without_waypoint_bypasses_grounding() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            {"type": "STOP"},
        ]
    )
    client = _Client(['{"motion":"stand","waypoints_2d":[]}'])

    AgentLoop(cast(Any, node), cast(Any, client)).run()

    command = agent_command_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert command.motion == "stand"
    assert command.target_xys == ()


def test_agent_preserves_stand_direction_for_in_place_turn() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"jpeg")),
            {"type": "STOP"},
        ]
    )
    client = _Client(['{"motion":"stand","direction":"left"}'])

    AgentLoop(cast(Any, node), cast(Any, client), command_mode="direction").run()

    command = agent_command_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert command.motion == "stand"
    assert command.direction == "left"


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
