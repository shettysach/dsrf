from typing import Any, cast

import numpy as np

from nodes.script_agent import ScriptAgentLoop
from script.tasks.arms_hold import ArmsHoldScript
from script.tasks.prompt import PromptScript
from script.tasks.push import PushScript
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


def test_push_script_emits_one_continuous_contact_and_push_motion() -> None:
    script = PushScript(
        prompt="reach and push with both palms", goal_position=(2.35, 0.0)
    )

    command = script.next_command(0)

    assert command is not None
    assert command.motion == "reach and push with both palms"
    np.testing.assert_allclose(command.target_xys, ((0.75, 0.0),))
    assert [target.name for target in command.end_effectors] == [
        "left_hand",
        "right_hand",
    ]
    left, right = command.end_effectors
    np.testing.assert_allclose(left.target_xyz, (1.20, 0.18, -0.16))
    np.testing.assert_allclose(right.target_xyz, (1.20, -0.18, -0.16))
    assert script.next_command(1) is None


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

    assert [output_id for output_id, _, _ in node.outputs] == ["command"]
    emitted = agent_command_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert emitted == command


def test_script_agent_can_send_without_an_observation() -> None:
    node = _Node([{"type": "STOP"}])

    ScriptAgentLoop(
        cast(Any, node),
        PushScript(prompt="touch the box with both palms"),
        start_immediately=True,
    ).run()

    assert [output_id for output_id, _, _ in node.outputs] == ["command"]


def test_arms_hold_script_constrains_both_hands() -> None:
    script = ArmsHoldScript(
        prompt="stand still with both arms extended straight forward at shoulder height, palms down, and hold the pose"
    )

    command = script.next_command(0)

    assert command is not None
    assert command.motion == script.prompt
    assert command.target_xys == ()
    assert command.end_effectors == (
        EndEffectorTarget("left_hand", (0.58, 0.16, 0.40)),
        EndEffectorTarget("right_hand", (0.58, -0.16, 0.40)),
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
