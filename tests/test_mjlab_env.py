from typing import Any, cast

import pytest
import torch

from sim.config import make_sim_env_cfg
from sim.env import MjlabEnv, _hand_object_contacts_from_buffers
from tasks.box_push import TASK as BOX_PUSH_TASK


def test_observation_camera_is_attached_to_torso() -> None:
    camera = make_sim_env_cfg().scene.sensors[0]

    assert camera.parent_body == "robot/torso_link"
    assert camera.pos == pytest.approx((-1.931852, 0.0, 0.517638), abs=1e-6)
    assert camera.fovy == 45.0


def test_box_push_starts_g1_directly_behind_the_box() -> None:
    cfg = make_sim_env_cfg(task=BOX_PUSH_TASK)

    robot = cfg.scene.entities["robot"]
    assert robot.init_state.pos == pytest.approx((0.80, 0.0, 0.76))


def test_mjlab_env_exposes_native_environment_for_mjlab_integrations() -> None:
    mjlab_env = cast(Any, MjlabEnv.__new__(MjlabEnv))
    native_env = object()
    mjlab_env._env = native_env

    assert mjlab_env.mjlab_env is native_env


def test_hand_object_contacts_are_filtered_with_torch_buffers() -> None:
    contacts = _hand_object_contacts_from_buffers(
        geom_pairs=torch.tensor(((10, 42), (11, 99), (10, 42), (10, 42))),
        world_ids=torch.tensor((0, 0, 1, 0)),
        contact_count=torch.tensor(3),
        hand_geom_ids={"left_hand": 10, "right_hand": 11},
        object_geom_ids={"box": frozenset((42,))},
    )

    assert contacts == {("left_hand", "box")}
