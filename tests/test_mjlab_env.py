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
