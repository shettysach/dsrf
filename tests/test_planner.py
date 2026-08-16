import pytest

from agent.planner import parse_kinematic_command


def test_kinematic_command_parses_valid_tool_arguments() -> None:
    assert parse_kinematic_command('{"motion":"stand","direction":"forward"}') == (
        "stand",
        None,
    )
    assert parse_kinematic_command('{"motion":"walk","direction":"left"}') == (
        "walk",
        "left",
    )


@pytest.mark.parametrize(
    "command",
    [
        '{"motion":"stand","direction":"left"}',
        '{"motion":"walk","direction":"up"}',
        '{"motion":"walk"}',
    ],
)
def test_kinematic_command_rejects_invalid_tool_arguments(command: str) -> None:
    with pytest.raises(ValueError):
        parse_kinematic_command(command)
