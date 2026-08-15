from __future__ import annotations

from io import StringIO

from agent.pi_debug import PiDebug


def test_pi_debug_formats_streaming_text_and_tool_calls() -> None:
    stream = StringIO()
    debug = PiDebug(True, stream=stream)

    debug.started(command_mode="direction", provider=None)
    debug.prompt(observation_id=3, completed_command='{"motion":"walk"}', retry=False)
    debug.event(
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_start", "contentIndex": 0},
        }
    )
    debug.event(
        {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "text_delta",
                "contentIndex": 0,
                "delta": "Moving left.",
            },
        }
    )
    debug.event({"type": "tool_execution_start", "toolName": "robot_action", "args": {"motion": "walk", "direction": "left"}})
    debug.event({"type": "agent_settled"})

    output = stream.getvalue()
    assert "started (provider=Pi default, command_mode=direction)" in output
    assert "prompting observation 3 (image attached" in output
    assert "assistant: Moving left." in output
    assert 'tool robot_action {"motion":"walk","direction":"left"}' in output
    assert "settled" in output


def test_pi_debug_is_silent_when_disabled() -> None:
    stream = StringIO()
    debug = PiDebug(False, stream=stream)

    debug.prompt(observation_id=0, completed_command=None, retry=False)
    debug.event({"type": "agent_settled"})

    assert stream.getvalue() == ""
