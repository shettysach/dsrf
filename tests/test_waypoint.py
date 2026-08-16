import math

import numpy as np
import pytest

from agent.constraint import parse_constraint_command
from shared.messages import GroundingRequest, ProjectionContext
from sim.waypoint import resolve_waypoint


def _floor_projection() -> ProjectionContext:
    size = 101
    camera_pos = np.array([0.0, 0.0, 1.0])
    forward = np.array([math.sqrt(0.5), 0.0, -math.sqrt(0.5)])
    up = np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    depth = np.full((size, size), np.nan, dtype=np.float32)
    for v in range(size):
        image_y = 1.0 - 2.0 * (v + 0.5) / size
        ray_z = forward[2] + up[2] * image_y * 0.5
        if ray_z < 0.0:
            depth[v, :] = -camera_pos[2] / ray_z
    return ProjectionContext(
        depth=depth,
        camera_pos_w=camera_pos,
        camera_forward_w=forward,
        camera_up_w=up,
        frustum_height=1.0,
        root_pos_w=np.zeros(3),
        root_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        near=0.01,
        far=100.0,
    )


def test_center_floor_pixel_resolves_forward() -> None:
    result = resolve_waypoint((500, 500), _floor_projection(), max_distance=10.0)
    assert result.pixel == (50, 50)
    assert result.target_xy[0] == pytest.approx(1.0, abs=0.03)
    assert result.target_xy[1] == pytest.approx(0.0, abs=0.01)


def test_left_image_pixel_maps_to_positive_robot_left() -> None:
    left = resolve_waypoint((300, 500), _floor_projection(), max_distance=10.0)
    right = resolve_waypoint((700, 500), _floor_projection(), max_distance=10.0)
    assert left.target_xy[1] > 0.0
    assert right.target_xy[1] < 0.0


def test_higher_floor_pixel_is_farther_than_lower_pixel() -> None:
    farther = resolve_waypoint((500, 400), _floor_projection(), max_distance=10.0)
    nearer = resolve_waypoint((500, 700), _floor_projection(), max_distance=10.0)
    assert farther.target_xy[0] > nearer.target_xy[0]


def test_depth_patch_uses_valid_median_and_clamps_horizon() -> None:
    projection = _floor_projection()
    projection.depth[48:53, 48:53] = np.nan
    projection.depth[49:52, 49:52] = math.sqrt(2.0)
    projection.depth[50, 50] = 50.0
    result = resolve_waypoint((500, 500), projection, max_distance=0.5)
    assert result.depth == pytest.approx(math.sqrt(2.0), rel=1e-5)
    assert math.hypot(*result.target_xy) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"motion":"walk","waypoints_2d":[[500]]}',
        '{"motion":"walk","waypoints_2d":[[500.0,500]]}',
    ],
)
def test_malformed_commands_fail(text: str) -> None:
    with pytest.raises(ValueError):
        parse_constraint_command(text)


def test_grounding_request_validates_coordinate_range() -> None:
    parsed = parse_constraint_command('{"motion":"walk","waypoints_2d":[[-1,500]]}')
    assert parsed.waypoints_2d
    with pytest.raises(ValueError, match=r"\[0,1000\]"):
        GroundingRequest(0, parsed.waypoints_2d)


def test_stand_parses_without_resolving_depth() -> None:
    command = parse_constraint_command('{"motion":"stand","waypoints_2d":[]}')
    assert command.motion == "stand"
    assert command.waypoints_2d == ()


def test_command_without_waypoints_defaults_to_no_grounding() -> None:
    command = parse_constraint_command('{"motion":"wave with the right hand"}')
    assert command.motion == "wave with the right hand"
    assert command.waypoints_2d == ()


def test_expressive_motion_prompt_can_optionally_have_a_waypoint() -> None:
    grounded = parse_constraint_command(
        '{"motion":"sidestep carefully toward the doorway","waypoints_2d":[[400,600],[600,500]]}'
    )
    ungrounded = parse_constraint_command(
        '{"motion":"wave with the right hand","waypoints_2d":[]}'
    )
    assert grounded.motion == "sidestep carefully toward the doorway"
    assert grounded.waypoints_2d == ((400, 600), (600, 500))
    assert ungrounded.waypoints_2d == ()


def test_invalid_depth_fails() -> None:
    projection = _floor_projection()
    projection.depth[:] = np.nan
    with pytest.raises(ValueError, match="No valid depth"):
        resolve_waypoint((500, 500), projection)
