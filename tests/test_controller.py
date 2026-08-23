from types import SimpleNamespace

import numpy as np
import pytest
import torch

from controller import ControlOutput, ExternalWrench, RobotState
from controller.g1_command import G1CommandTransform
from controller.reference import MotionReference
from controller.sonic import SonicController
from controller.virtual_forces import VirtualForcesController
from shared.config import VirtualForcesConfig
from shared.messages import MotionChunk


def _state() -> RobotState:
    return RobotState(
        root_pos_w=torch.tensor([3.0, 4.0, 0.8]),
        root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        root_lin_vel_w=torch.zeros(3),
        root_ang_vel_w=torch.zeros(3),
        root_ang_vel_b=torch.zeros(3),
        projected_gravity_b=torch.tensor([0.0, 0.0, -1.0]),
        joint_pos=torch.zeros(29),
        joint_vel=torch.zeros(29),
    )


def _motion(frames: int = 2) -> MotionChunk:
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 2] = 0.8
    if frames > 1:
        qpos[1, 0] = 0.1
        qpos[1, 7:] = np.arange(29)
    return MotionChunk(0, "test motion", qpos)


def test_g1_command_transform_round_trip() -> None:
    default = torch.linspace(-0.3, 0.3, 29)
    scale = torch.linspace(0.01, 0.1, 29)
    transform = G1CommandTransform(default, scale)
    action = torch.linspace(-1.0, 1.0, 29)

    torch.testing.assert_close(transform.encode(transform.decode(action)), action)
    target = torch.linspace(-0.5, 0.5, 29)
    torch.testing.assert_close(transform.decode(transform.encode(target)), target)


def test_sonic_controller_preserves_raw_action_through_physical_boundary() -> None:
    transform = G1CommandTransform(torch.linspace(-0.3, 0.3, 29), torch.ones(29) * 0.1)
    raw_action = torch.linspace(-1.0, 1.0, 29)
    controller = SonicController.__new__(SonicController)
    controller.command_transform = transform
    controller.policy = SimpleNamespace(
        infer=lambda state: (raw_action.unsqueeze(0), True)
    )

    output = controller.act(_state())

    torch.testing.assert_close(transform.encode(output.joint_target), raw_action)
    assert output.completed


def test_control_contract_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="joint_target"):
        ControlOutput(torch.zeros((1, 29)))
    with pytest.raises(ValueError, match="force_w"):
        ExternalWrench("pelvis", torch.zeros(2), torch.zeros(3))


def test_shared_reference_accepts_one_frame() -> None:
    reference = MotionReference("cpu")
    reference.load(_motion(1), _state().root_pos_w, _state().root_quat_w)

    frame = reference.current()

    torch.testing.assert_close(frame.root_lin_vel_w, torch.zeros(3))
    torch.testing.assert_close(frame.root_ang_vel_w, torch.zeros(3))
    assert reference.advance()


def test_virtual_forces_controller_tracks_reference_without_assistance() -> None:
    controller = VirtualForcesController(VirtualForcesConfig())
    controller.load_motion(_motion(), _state())

    first = controller.act(_state())
    second = controller.act(_state())

    torch.testing.assert_close(first.joint_target, torch.zeros(29))
    torch.testing.assert_close(second.joint_target, torch.arange(29).float())
    assert first.external_wrenches == ()
    assert not first.completed
    assert second.completed


def test_virtual_force_is_bounded() -> None:
    config = VirtualForcesConfig(
        assistance_enabled=True,
        position_kp=1000.0,
        position_kd=0.0,
        force_limit=10.0,
    )
    controller = VirtualForcesController(config)
    motion = _motion()
    motion.qpos[0, :3] = [0.0, 0.0, 2.0]
    controller.load_motion(motion, _state())

    output = controller.act(_state())

    assert len(output.external_wrenches) == 1
    assert float(
        torch.linalg.vector_norm(output.external_wrenches[0].force_w)
    ) == pytest.approx(10.0)
