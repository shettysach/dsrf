from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from tasks.box_push import TASK as BOX_PUSH_TASK
from tasks.box_push.scene import (
    make_box_push_entity_cfg,
)
from tasks.box_push.settings import BoxPushSettings

from sim.config import make_sim_env_cfg
from sim.env import MjlabEnv, SokobanMotionEvents, _hand_object_contacts_from_buffers
from sim.viewer import ViserSimViewer


def test_observation_camera_is_attached_to_torso() -> None:
    camera = make_sim_env_cfg().scene.sensors[0]

    assert camera.parent_body == "robot/torso_link"
    assert camera.pos == pytest.approx((-1.931852, 0.0, 0.517638), abs=1e-6)
    assert camera.fovy == 45.0


def test_viser_viewer_syncs_only_when_a_browser_is_connected() -> None:
    viewer = object.__new__(ViserSimViewer)
    viewer._server = Mock()  # type: ignore[attr-defined]
    viewer.sync_env_to_viewer = Mock()  # type: ignore[method-assign]

    viewer._server.get_clients.return_value = {}  # type: ignore[attr-defined]
    viewer.sync()
    viewer.sync_env_to_viewer.assert_not_called()

    viewer._server.get_clients.return_value = {1: object()}  # type: ignore[attr-defined]
    viewer.sync()
    viewer.sync_env_to_viewer.assert_called_once()


def test_box_push_starts_g1_directly_behind_the_box() -> None:
    cfg = make_sim_env_cfg(task=BOX_PUSH_TASK)

    robot = cfg.scene.entities["robot"]
    assert robot.init_state.pos == pytest.approx((0.0, 0.0, 0.76))


def test_box_push_box_has_a_wide_stable_footprint() -> None:
    box = make_box_push_entity_cfg()
    settings = BoxPushSettings()

    assert settings.half_size == pytest.approx((0.50, 0.50, 0.65))
    assert box.init_state.pos == pytest.approx((3.0, 0.0, 0.65))


def test_box_push_box_starts_entirely_before_the_goal() -> None:
    settings = BoxPushSettings()
    initial_box_front = settings.box_x + settings.half_size[0]
    goal_back = settings.goal_x - settings.goal_half_size

    assert goal_back - initial_box_front == pytest.approx(1.85)
    assert settings.goal_half_size >= settings.half_size[0]
    assert settings.goal_half_size >= settings.half_size[1]


def test_box_push_uses_a_world_overview_with_box_assistance() -> None:
    camera = make_sim_env_cfg(task=BOX_PUSH_TASK).scene.sensors[0]

    assert BOX_PUSH_TASK.virtual_force_objects == ("box",)
    assert camera.parent_body is None
    assert camera.pos == pytest.approx((0.0, -6.0, 5.0))
    assert camera.fovy == 65.0


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


def test_sokoban_motion_feedback_indexes_batched_geometry_positions() -> None:
    """Sokoban geometry IDs index the geom axis, after MJLab's batch axis."""

    positions = torch.zeros((1, 80, 3))
    positions[0, 75, :2] = torch.tensor((2.25, -1.5))
    data = type("Data", (), {"geom_xpos": positions})()
    events = SokobanMotionEvents(
        robot_geom_ids=frozenset(),
        wall_geom_ids=frozenset(),
        box_geom_ids=(75,),
        initial_box_centers=np.array(((2.0, -1.5),)),
    )

    assert events.feedback(data) == "Box 1 pushed: Δx=+0.25 m, Δy=+0.00 m."
