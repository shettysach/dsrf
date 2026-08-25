from typing import Any, cast

import pytest

from sim.config import make_sim_env_cfg
from sim.env import MjlabEnv


def test_observation_camera_is_attached_to_torso() -> None:
    camera = make_sim_env_cfg().scene.sensors[0]

    assert camera.parent_body == "robot/torso_link"
    assert camera.pos == pytest.approx((-1.931852, 0.0, 0.517638), abs=1e-6)
    assert camera.fovy == 45.0


def test_mjlab_env_exposes_native_environment_for_mjlab_integrations() -> None:
    mjlab_env = cast(Any, MjlabEnv.__new__(MjlabEnv))
    native_env = object()
    mjlab_env._env = native_env

    assert mjlab_env.mjlab_env is native_env
