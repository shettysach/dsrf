from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from controller import ControlOutput, ExternalWrench
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


class _Robot:
    def __init__(self) -> None:
        self.writes: list[tuple[torch.Tensor, torch.Tensor]] = []

    def write_external_wrench_to_sim(self, forces, torques) -> None:
        self.writes.append((forces.clone(), torques.clone()))


def _wrench_env() -> MjlabEnv:
    simulation = MjlabEnv.__new__(MjlabEnv)
    simulation._robot = _Robot()
    simulation._body_ids = {"pelvis": 0, "torso_link": 1}
    simulation._wrench_forces = torch.zeros((1, 2, 3))
    simulation._wrench_torques = torch.zeros((1, 2, 3))
    return simulation


def test_external_wrench_is_applied_then_cleared() -> None:
    simulation = _wrench_env()
    wrench = ExternalWrench(
        "pelvis",
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
    )

    simulation._apply_wrenches(
        ControlOutput(torch.zeros(29), external_wrenches=(wrench,))
    )
    simulation._apply_wrenches(ControlOutput(torch.zeros(29)))

    robot = cast(_Robot, simulation._robot)
    torch.testing.assert_close(robot.writes[0][0][0, 0], wrench.force_w)
    torch.testing.assert_close(robot.writes[0][1][0, 0], wrench.torque_w)
    assert not bool(robot.writes[1][0].any())
    assert not bool(robot.writes[1][1].any())


def test_external_wrench_rejects_unknown_and_duplicate_bodies() -> None:
    simulation = _wrench_env()
    wrench = ExternalWrench("pelvis", torch.zeros(3), torch.zeros(3))
    unknown = ExternalWrench("head", torch.zeros(3), torch.zeros(3))

    with pytest.raises(ValueError, match="Unknown"):
        simulation._apply_wrenches(
            ControlOutput(torch.zeros(29), external_wrenches=(unknown,))
        )
    with pytest.raises(ValueError, match="Duplicate"):
        simulation._apply_wrenches(
            ControlOutput(torch.zeros(29), external_wrenches=(wrench, wrench))
        )
