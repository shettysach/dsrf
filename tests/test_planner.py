import pytest

from motion_gen.kinematic_planner.parser import (
    KinematicPlannerCommand,
    parse_kinematic_planner_command,
)


def test_kinematic_command_parses_valid_tool_arguments() -> None:
    assert parse_kinematic_planner_command(
        '{"motion":"stand","direction":"left"}'
    ) == KinematicPlannerCommand("stand", "left", ())
    assert parse_kinematic_planner_command(
        '{"motion":"walk","direction":"left"}'
    ) == KinematicPlannerCommand("walk", "left", ())
    assert parse_kinematic_planner_command(
        '{"motion":"walk","waypoints_2d":[[300,600],[600,500]]}'
    ) == KinematicPlannerCommand("walk", None, ((300, 600), (600, 500)))


@pytest.mark.parametrize(
    "command",
    [
        '{"motion":"stand","direction":"up"}',
        '{"motion":"walk","direction":"up"}',
        '{"motion":"walk"}',
        '{"motion":"walk","direction":"forward","waypoints_2d":[[500,500]]}',
    ],
)
def test_kinematic_command_rejects_invalid_tool_arguments(command: str) -> None:
    with pytest.raises(ValueError):
        parse_kinematic_planner_command(command)
