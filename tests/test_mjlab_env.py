from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from sim.env import MjlabEnv


def test_render_keeps_camera_azimuth_fixed() -> None:
    renderer = SimpleNamespace(
        _cam=SimpleNamespace(azimuth=10.0),
        render=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
    )
    offline = SimpleNamespace(
        renderer=renderer,
        update=lambda data, debug_vis_callback: None,
    )
    simulation = cast(Any, MjlabEnv.__new__(MjlabEnv))
    simulation.cuda_stream = None
    simulation._env = SimpleNamespace(
        _offline_renderer=offline,
        sim=SimpleNamespace(data=object()),
    )

    simulation.render()

    assert renderer._cam.azimuth == 10.0


def test_fall_detection_does_not_reset_an_upright_squat() -> None:
    simulation = cast(Any, MjlabEnv.__new__(MjlabEnv))
    simulation.robot_state = lambda: SimpleNamespace(
        root_pos_w=np.array((0.0, 0.0, 0.44)),
        root_quat_w=np.array((1.0, 0.0, 0.0, 0.0)),
    )

    assert simulation.fall_reason() is None


def test_fall_detection_reports_collapsed_or_tipped_torso() -> None:
    simulation = cast(Any, MjlabEnv.__new__(MjlabEnv))
    simulation.robot_state = lambda: SimpleNamespace(
        root_pos_w=np.array((0.0, 0.0, 0.20)),
        root_quat_w=np.array((1.0, 0.0, 0.0, 0.0)),
    )
    assert "too low" in simulation.fall_reason()

    simulation.robot_state = lambda: SimpleNamespace(
        root_pos_w=np.array((0.0, 0.0, 0.70)),
        root_quat_w=np.array((0.5, 0.866, 0.0, 0.0)),
    )
    assert "tipped over" in simulation.fall_reason()
