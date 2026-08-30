from typing import Any, cast

import numpy as np

from nodes.script_agent import ScriptAgentLoop
from script.tasks.push import PushScript
from shared.arrow import agent_command_from_arrow, observation_to_arrow
from shared.messages import VisualObservation


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


def _observation_event(observation: VisualObservation) -> dict[str, object]:
    value, metadata = observation_to_arrow(observation)
    return {
        "type": "INPUT",
        "id": "observation",
        "value": value,
        "metadata": metadata,
    }


def test_push_script_emits_one_symmetric_two_palm_command() -> None:
    script = PushScript()

    command = script.next_command(0)

    assert command is not None
    assert command.motion == "reach forward with both hands and push the box"
    assert command.target_xys == ()
    assert [target.name for target in command.end_effectors] == [
        "left_hand",
        "right_hand",
    ]
    left, right = command.end_effectors
    np.testing.assert_allclose(left.target_xyz, (1.52, 0.16, 0.55))
    np.testing.assert_allclose(right.target_xyz, (1.52, -0.16, 0.55))
    assert script.next_command(1) is None


def test_script_agent_sends_commands_through_the_normal_agent_channel() -> None:
    command = PushScript().next_command(0)
    assert command is not None
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"first")),
            _observation_event(VisualObservation(1, command.text, b"second")),
            {"type": "STOP"},
        ]
    )

    ScriptAgentLoop(cast(Any, node), PushScript()).run()

    assert [output_id for output_id, _, _ in node.outputs] == ["command"]
    emitted = agent_command_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert emitted == command
