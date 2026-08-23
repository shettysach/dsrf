import math

import pytest
import torch

from motion_gen.ardy.parser import parse_ardy_command
from shared.messages import GroundingRequest
from sim.camera import ProjectionContext
from sim.grounding import resolve_end_effector, resolve_waypoint


def _floor_projection() -> ProjectionContext:
    size = 101
    camera_pos = torch.tensor([0.0, 0.0, 1.0])
    forward = torch.tensor([math.sqrt(0.5), 0.0, -math.sqrt(0.5)])
    up = torch.tensor([math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    right = torch.cross(forward, up, dim=0)
    rotation = torch.stack((right, up, -forward), dim=1)
    depth = torch.full((size, size), math.nan)
    for v in range(size):
        image_y = 1.0 - 2.0 * (v + 0.5) / size
        ray_z = forward[2] + up[2] * image_y * 0.5
        if ray_z < 0.0:
            depth[v, :] = -camera_pos[2] / ray_z
    return ProjectionContext(
        depth=depth,
        camera_pos_w=camera_pos,
        camera_rotation_w=rotation,
        fovy_rad=2.0 * math.atan(0.5),
        root_pos_w=torch.zeros(3),
        root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
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
    projection.depth[48:53, 48:53] = math.nan
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
        parse_ardy_command(text)


def test_grounding_request_validates_coordinate_range() -> None:
    parsed = parse_ardy_command('{"motion":"walk","waypoints_2d":[[-1,500]]}')
    assert parsed.waypoints_2d
    with pytest.raises(ValueError, match=r"\[0,1000\]"):
        GroundingRequest(0, parsed.waypoints_2d)


def test_stand_parses_without_resolving_depth() -> None:
    command = parse_ardy_command('{"motion":"stand","waypoints_2d":[]}')
    assert command.motion == "stand"
    assert command.waypoints_2d == ()


def test_command_without_waypoints_defaults_to_no_grounding() -> None:
    command = parse_ardy_command('{"motion":"wave with the right hand"}')
    assert command.motion == "wave with the right hand"
    assert command.waypoints_2d == ()


def test_end_effector_command_parses_without_waypoints() -> None:
    command = parse_ardy_command(
        '{"motion":"reach for the cup","end_effectors":'
        '[{"name":"right_hand","target_2d":[600,400]}]}'
    )
    assert command.waypoints_2d == ()
    assert [(target.name, target.target_2d) for target in command.end_effectors] == [
        ("right_hand", (600, 400))
    ]


def test_foot_end_effector_command_parses_without_waypoints() -> None:
    command = parse_ardy_command(
        '{"motion":"step onto the platform","end_effectors":'
        '[{"name":"left_foot","target_2d":[500,700]}]}'
    )
    assert [(target.name, target.target_2d) for target in command.end_effectors] == [
        ("left_foot", (500, 700))
    ]


@pytest.mark.parametrize(
    "text",
    [
        '{"motion":"reach","end_effectors":null}',
        '{"motion":"reach","end_effectors":[{"name":"head","target_2d":[500,500]}]}',
    ],
)
def test_invalid_end_effector_commands_fail(text: str) -> None:
    with pytest.raises(ValueError):
        parse_ardy_command(text)


def test_command_can_combine_waypoints_and_end_effectors() -> None:
    command = parse_ardy_command(
        '{"motion":"walk to the box and reach","waypoints_2d":[[500,700]],'
        '"end_effectors":[{"name":"left_hand","target_2d":[550,400]}]}'
    )
    assert command.waypoints_2d == ((500, 700),)
    assert command.end_effectors[0].name == "left_hand"


def test_expressive_motion_prompt_can_optionally_have_a_waypoint() -> None:
    grounded = parse_ardy_command(
        '{"motion":"sidestep carefully toward the doorway","waypoints_2d":[[400,600],[600,500]]}'
    )
    ungrounded = parse_ardy_command(
        '{"motion":"wave with the right hand","waypoints_2d":[]}'
    )
    assert grounded.motion == "sidestep carefully toward the doorway"
    assert grounded.waypoints_2d == ((400, 600), (600, 500))
    assert ungrounded.waypoints_2d == ()


def test_invalid_depth_fails() -> None:
    projection = _floor_projection()
    projection.depth[:] = math.nan
    with pytest.raises(ValueError, match="No valid depth"):
        resolve_waypoint((500, 500), projection)


def test_end_effector_pixel_resolves_to_local_3d_target() -> None:
    projection = ProjectionContext(
        depth=torch.full((101, 101), 0.5),
        camera_pos_w=torch.tensor([0.0, 0.0, 0.8]),
        camera_rotation_w=torch.tensor(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        fovy_rad=2.0 * math.atan(0.5),
        root_pos_w=torch.zeros(3),
        root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        near=0.01,
        far=100.0,
    )

    result = resolve_end_effector("right_hand", (500, 500), projection)

    assert result.pixel == (50, 50)
    assert result.target_xyz == pytest.approx((0.5, 0.0, 0.8), abs=0.01)
