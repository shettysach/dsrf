from typing import Any, cast

import pytest
import torch
from tasks.box_push import TASK as BOX_PUSH_TASK
from tasks.box_push.scene import (
    BOX_HALF_SIZE,
    BOX_START,
    DEFAULT_GOAL_X,
    GOAL_HALF_SIZE,
    make_box_push_entity_cfg,
)

from sim.config import make_sim_env_cfg
from sim.env import MjlabEnv, _hand_object_contacts_from_buffers


def test_observation_camera_is_attached_to_torso() -> None:
    camera = make_sim_env_cfg().scene.sensors[0]

    assert camera.parent_body == "robot/torso_link"
    assert camera.pos == pytest.approx((-1.931852, 0.0, 0.517638), abs=1e-6)
    assert camera.fovy == 45.0


def test_box_push_starts_g1_directly_behind_the_box() -> None:
    cfg = make_sim_env_cfg(task=BOX_PUSH_TASK)

    robot = cfg.scene.entities["robot"]
    assert robot.init_state.pos == pytest.approx((0.75, 0.0, 0.76))


def test_box_push_box_has_a_wide_stable_footprint() -> None:
    box = make_box_push_entity_cfg()

    assert BOX_HALF_SIZE == pytest.approx((0.25, 0.45, 0.50))
    assert box.init_state.pos == pytest.approx((1.40, 0.0, 0.50))


def test_box_push_box_starts_entirely_before_the_goal() -> None:
    initial_box_front = BOX_START[0] + BOX_HALF_SIZE[0]
    goal_back = DEFAULT_GOAL_X - GOAL_HALF_SIZE[0]

    assert goal_back - initial_box_front == pytest.approx(0.10)
    assert GOAL_HALF_SIZE[0] >= BOX_HALF_SIZE[0]
    assert GOAL_HALF_SIZE[1] >= BOX_HALF_SIZE[1]


def test_box_push_uses_an_egocentric_observation_camera() -> None:
    camera = make_sim_env_cfg(task=BOX_PUSH_TASK).scene.sensors[0]

    assert BOX_PUSH_TASK.observation_camera.egocentric is True
    assert camera.parent_body == "robot/torso_link"
    assert camera.pos == pytest.approx((0.0, 0.0, 0.43))
    assert camera.fovy == 90.0


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
