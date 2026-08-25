import pytest
import torch

from sim.config import make_sim_env_cfg
from sim.env import MjlabEnv


def test_observation_camera_is_attached_to_torso() -> None:
    camera = make_sim_env_cfg().scene.sensors[0]

    assert camera.parent_body == "robot/torso_link"
    assert camera.pos == pytest.approx((-1.931852, 0.0, 0.517638), abs=1e-6)
    assert camera.fovy == 45.0


def test_step_rejects_non_sonic_action_shape() -> None:
    simulation = MjlabEnv.__new__(MjlabEnv)
    simulation.num_envs = 1

    with pytest.raises(ValueError, match="SONIC action"):
        simulation.step(torch.zeros(29))
