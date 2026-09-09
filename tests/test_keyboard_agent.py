from typing import Any, cast

from nodes.keyboard_agent import _DIRECTIONS, KeyboardSokobanAgentLoop
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
    return {"type": "INPUT", "id": "observation", "value": value, "metadata": metadata}


def test_keyboard_agent_maps_wasd_and_hjkl_to_planner_directions() -> None:
    node = _Node(
        [
            _observation_event(VisualObservation(0, None, b"first")),
            _observation_event(
                VisualObservation(1, '{"motion":"walk","direction":"left"}', b"second")
            ),
            _observation_event(VisualObservation(2, "finished", b"third")),
            {"type": "STOP"},
        ]
    )
    keys = iter(("h", "f"))

    KeyboardSokobanAgentLoop(cast(Any, node), read_key=lambda: next(keys)).run()

    commands = [
        agent_command_from_arrow(value, cast(Any, kwargs["metadata"]))
        for _, value, kwargs in node.outputs
    ]
    assert [(command.motion, command.direction, command.terminal) for command in commands] == [
        ("walk", "left", False),
        ("stand", None, True),
    ]
    assert commands[1].text == "finished"


def test_wasd_and_hjkl_have_matching_direction_mappings() -> None:
    assert _DIRECTIONS == {
        "w": "forward",
        "k": "forward",
        "s": "backward",
        "j": "backward",
        "a": "left",
        "h": "left",
        "d": "right",
        "l": "right",
    }
