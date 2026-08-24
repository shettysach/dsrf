from types import SimpleNamespace

import numpy as np
import pytest
import torch

from controller import ControlOutput, ExternalWrench, RobotState
from controller.direct import DirectController
from controller.g1_command import G1CommandTransform
from controller.reference import MotionReference
from controller.sonic import SonicController
from shared.config import DirectConfig
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


def _motion() -> MotionChunk:
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 2] = 0.8
    qpos[1, 7:] = np.arange(29)
    return MotionChunk(0, "test motion", qpos)


def test_control_contract_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="joint_target"):
        ControlOutput(torch.zeros((1, 29)))
    with pytest.raises(ValueError, match="force_w"):
        ExternalWrench("pelvis", torch.zeros(2), torch.zeros(3))


def test_direct_controller_is_pure_joint_reference_tracking() -> None:
    controller = DirectController(DirectConfig())
    controller.load_motion(_motion(), _state())
    expected = controller.reference.current()

    output = controller.act(_state())

    torch.testing.assert_close(output.joint_target, expected.joint_pos)
    torch.testing.assert_close(output.joint_velocity_target, expected.joint_vel)
    assert output.external_wrenches == ()


def test_reference_starts_at_live_root_pose() -> None:
    reference = MotionReference("cpu")
    motion = _motion()
    motion.qpos[0, :7] = [1.0, 2.0, 1.3, 0.5, 0.5, 0.5, 0.5]
    state = _state()
    reference.load(motion, state.root_pos_w, state.root_quat_w)
    frame = reference.current()
    torch.testing.assert_close(frame.root_pos_w, state.root_pos_w)
    torch.testing.assert_close(frame.root_quat_w, state.root_quat_w)


def test_sonic_controller_preserves_raw_action() -> None:
    transform = G1CommandTransform(torch.zeros(29), torch.ones(29) * 0.1)
    raw_action = torch.linspace(-1.0, 1.0, 29)
    controller = SonicController.__new__(SonicController)
    controller.command_transform = transform
    controller.policy = SimpleNamespace(infer=lambda state: (raw_action[None], True))
    output = controller.act(_state())
    torch.testing.assert_close(transform.encode(output.joint_target), raw_action)
    assert output.completed
