import math

import numpy as np
import pytest

from shared.geometry import (
    local_xy_to_world,
    world_xy_to_local,
    yaw_from_quat_wxyz,
)


def test_yaw_from_wxyz_quaternion() -> None:
    yaw = math.pi / 2.0
    quaternion = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])

    assert yaw_from_quat_wxyz(quaternion) == pytest.approx(yaw)


def test_local_and_world_xy_transforms_are_inverses() -> None:
    local = np.array([1.2, -0.4])
    yaw = 0.7

    world = local_xy_to_world(float(local[0]), float(local[1]), yaw)

    np.testing.assert_allclose(world_xy_to_local(world, yaw), local)


def test_yaw_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="norm"):
        yaw_from_quat_wxyz(np.zeros(4))
