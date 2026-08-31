from typing import Any, cast

import numpy as np

from nodes.script_agent import ScriptAgentLoop
from script.tasks.prompt import PromptScript
from script.tasks.push import PushScript
from script.tasks.right_kick import RightKickScript
from shared.arrow import agent_command_from_arrow, observation_to_arrow
from shared.messages import EndEffectorTarget, VisualObservation


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


def test_push_script_reaches_then_walks_the_box_to_the_goal() -> None:
    script = PushScript(prompt="touch the box with both palms")

    command = script.next_command(0)

    assert command is not None
    assert command.motion == "extend both arms straight forward and hold them there"
    assert command.target_xys == ()
    assert [target.name for target in command.end_effectors] == [
        "left_hand",
        "right_hand",
    ]
    left, right = command.end_effectors
    np.testing.assert_allclose(left.target_xyz, (0.60, 0.16, 0.29))
    np.testing.assert_allclose(right.target_xyz, (0.60, -0.16, 0.29))

    push = script.next_command(1)

    assert push is not None
    assert push.motion == "walk forward while pushing the box with both palms"
    np.testing.assert_allclose(push.target_xys, ((3.45, 0.0),))
    np.testing.assert_allclose(push.end_effectors[0].target_xyz, (4.05, 0.16, 0.29))
    np.testing.assert_allclose(push.end_effectors[1].target_xyz, (4.05, -0.16, 0.29))
    assert script.next_command(2) is None


def test_script_agent_sends_commands_through_the_normal_agent_channel() -> None:
    command = PushScript(prompt="touch the box with both palms").next_command(0)
    assert command is not None
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"first")),
            _observation_event(VisualObservation(1, command.text, b"second")),
            {"type": "STOP"},
        ]
    )

    ScriptAgentLoop(
        cast(Any, node), PushScript(prompt="touch the box with both palms")
    ).run()

    assert [output_id for output_id, _, _ in node.outputs] == ["command", "command"]
    emitted = agent_command_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert emitted == command
    push = agent_command_from_arrow(
        node.outputs[1][1], cast(Any, node.outputs[1][2]["metadata"])
    )
    assert push == PushScript(prompt="touch the box with both palms").next_command(1)


def test_script_agent_can_send_without_an_observation() -> None:
    node = _Node([{"type": "STOP"}])

    ScriptAgentLoop(
        cast(Any, node),
        PushScript(prompt="touch the box with both palms"),
        start_immediately=True,
    ).run()

    assert [output_id for output_id, _, _ in node.outputs] == ["command"]


def test_right_kick_script_constrains_only_the_right_foot() -> None:
    script = RightKickScript(prompt="perform a high kick forward with the right leg")

    command = script.next_command(0)

    assert command is not None
    assert command.motion == script.prompt
    assert command.target_xys == ()
    assert command.end_effectors == (
        EndEffectorTarget("right_foot", (0.65, -0.15, 0.65)),
    )
    assert script.next_command(1) is None


def test_prompt_script_emits_one_unconstrained_motion() -> None:
    prompt = "extend both arms straight forward and hold them there"

    command = PromptScript(prompt).next_command(0)

    assert command is not None
    assert command.text == command.motion == prompt
    assert command.target_xys == ()
    assert command.end_effectors == ()
    assert PromptScript(prompt).next_command(1) is None
