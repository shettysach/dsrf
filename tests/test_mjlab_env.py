from typing import cast

import pytest
import torch
from mjlab.actuator import BuiltinPdActuatorCfg

from controller import ControlOutput, ExternalWrench
from controller.g1_command import G1CommandTransform
from sim.config import make_sim_env_cfg
from sim.env import MjlabEnv


def test_observation_camera_is_attached_to_torso() -> None:
    camera = make_sim_env_cfg().scene.sensors[0]

    assert camera.parent_body == "robot/torso_link"
    assert camera.pos == pytest.approx((-1.931852, 0.0, 0.517638), abs=1e-6)
    assert camera.fovy == 45.0


def test_pd_configuration_adds_velocity_actions_and_pd_actuators() -> None:
    cfg = make_sim_env_cfg(control_mode="pd")

    assert tuple(cfg.actions) == ("joint_position", "joint_velocity")
    position = cfg.actions["joint_position"]
    velocity = cfg.actions["joint_velocity"]
    assert position.actuator_names == (".*",)
    assert velocity.actuator_names == (".*",)
    assert velocity.scale == 1.0
    assert not velocity.use_default_offset
    assert all(
        isinstance(actuator, BuiltinPdActuatorCfg)
        for actuator in cfg.scene.entities["robot"].articulation.actuators
    )


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


def _action_env(control_mode: str) -> MjlabEnv:
    simulation = MjlabEnv.__new__(MjlabEnv)
    simulation.control_mode = control_mode
    simulation.command_transform = G1CommandTransform(
        torch.linspace(-0.3, 0.3, 29), torch.full((29,), 0.1)
    )
    return simulation


def test_pd_control_action_encodes_position_and_appends_physical_velocity() -> None:
    simulation = _action_env("pd")
    target = torch.linspace(-0.5, 0.5, 29)
    velocity = torch.linspace(-1.0, 1.0, 29)

    action = simulation._control_to_action(
        ControlOutput(target, joint_velocity_target=velocity)
    )

    assert action.shape == (1, 58)
    torch.testing.assert_close(
        action[0, :29], simulation.command_transform.encode(target)
    )
    torch.testing.assert_close(action[0, 29:], velocity)


def test_pd_control_action_uses_zero_velocity_when_unset() -> None:
    simulation = _action_env("pd")

    action = simulation._control_to_action(ControlOutput(torch.zeros(29)))

    torch.testing.assert_close(action[0, 29:], torch.zeros(29))


def test_position_control_rejects_velocity_targets() -> None:
    simulation = _action_env("position")

    with pytest.raises(ValueError, match="Position control mode"):
        simulation._control_to_action(
            ControlOutput(torch.zeros(29), joint_velocity_target=torch.zeros(29))
        )


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
