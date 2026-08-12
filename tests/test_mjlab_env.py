from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from sim.env import MjlabEnv, _is_robot_task_collision


def test_collision_notification_requires_robot_and_task_obstacle() -> None:
    assert _is_robot_task_collision(
        "robot/left_foot1_collision", "sokoban_left_wall_collision"
    )
    assert not _is_robot_task_collision("terrain", "robot/left_foot1_collision")
    assert not _is_robot_task_collision(
        "sokoban_box_1_pushable", "sokoban_left_wall_collision"
    )
    assert not _is_robot_task_collision(
        "robot/left_foot1_collision", "sokoban_box_1_pushable"
    )


def test_render_keeps_camera_azimuth_fixed() -> None:
    renderer = SimpleNamespace(_cam=SimpleNamespace(azimuth=10.0))
    simulation = cast(Any, MjlabEnv.__new__(MjlabEnv))
    simulation.cuda_stream = None
    simulation._env = SimpleNamespace(
        _offline_renderer=renderer,
        render=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
    )

    simulation.render()

    assert renderer._cam.azimuth == 10.0


def test_render_demo_uses_offscreen_renderer_bridge_and_restores_camera() -> None:
    camera = SimpleNamespace(distance=2.0, azimuth=10.0, elevation=-15.0)
    renderer = SimpleNamespace(
        _cam=camera,
        update=lambda data: setattr(renderer, "updated_with", data),
        render=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
    )
    simulation = cast(Any, MjlabEnv.__new__(MjlabEnv))
    simulation.cuda_stream = None
    data = object()
    simulation._env = SimpleNamespace(
        _offline_renderer=renderer,
        sim=SimpleNamespace(data=data),
    )

    image = simulation.render_demo_rgb()

    assert image.shape == (1, 1, 3)
    assert renderer.updated_with is data
    assert (camera.distance, camera.azimuth, camera.elevation) == (2.0, 10.0, -15.0)
