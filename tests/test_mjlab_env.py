from types import SimpleNamespace
from typing import Any, cast

import pytest

from sim.config import make_sim_env_cfg
from sim.env import MjlabEnv


def test_observation_camera_is_attached_to_torso() -> None:
    camera = make_sim_env_cfg().scene.sensors[0]

    assert camera.parent_body == "robot/torso_link"
    assert camera.pos == pytest.approx((-1.931852, 0.0, 0.517638), abs=1e-6)
    assert camera.fovy == 45.0


def test_viewer_environment_properties_forward_to_mjlab() -> None:
    mjlab_env = cast(Any, MjlabEnv.__new__(MjlabEnv))
    cfg = object()
    device = object()
    mjlab_env._env = SimpleNamespace(cfg=cfg, device=device, num_envs=1)

    assert mjlab_env.cfg is cfg
    assert mjlab_env.device is device
    assert mjlab_env.num_envs == 1
