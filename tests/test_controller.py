from types import SimpleNamespace

import numpy as np
import pytest
import torch

from controller import ControlOutput, ExternalWrench, RobotState
from controller.direct import DirectController
from controller.g1_command import G1CommandTransform
from controller.reference import MotionReference, ReferenceFrame
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
    assert output.joint_velocity_target is None


def test_control_contract_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="joint_target"):
        ControlOutput(torch.zeros((1, 29)))
    with pytest.raises(ValueError, match="force_w"):
        ExternalWrench("pelvis", torch.zeros(2), torch.zeros(3))
    with pytest.raises(ValueError, match="joint_velocity_target"):
        ControlOutput(torch.zeros(29), joint_velocity_target=torch.zeros((1, 29)))


def test_control_contract_accepts_velocity_target_without_changing_positional_fields() -> (
    None
):
    output = ControlOutput(torch.zeros(29), True, (), torch.ones(29))

    assert output.completed
    torch.testing.assert_close(output.joint_velocity_target, torch.ones(29))


def test_shared_reference_accepts_one_frame() -> None:
    reference = MotionReference("cpu")
    reference.load(_motion(1), _state().root_pos_w, _state().root_quat_w)

    frame = reference.current()

    torch.testing.assert_close(frame.root_lin_vel_w, torch.zeros(3))
    torch.testing.assert_close(frame.root_ang_vel_w, torch.zeros(3))
    assert reference.advance()


def test_shared_reference_derives_known_joint_velocity() -> None:
    reference = MotionReference("cpu")
    motion = _motion()
    motion.qpos[0, 7:] = 0.0
    motion.qpos[1, 7:] = 0.02
    reference.load(motion, _state().root_pos_w, _state().root_quat_w)

    first = reference.current()
    reference.advance()
    second = reference.current()

    torch.testing.assert_close(first.joint_vel, torch.ones(29))
    torch.testing.assert_close(second.joint_vel, torch.ones(29))


def _reference(**overrides: torch.Tensor) -> ReferenceFrame:
    values = {
        "root_pos_w": _state().root_pos_w,
        "root_quat_w": _state().root_quat_w,
        "root_lin_vel_w": _state().root_lin_vel_w,
        "root_ang_vel_w": _state().root_ang_vel_w,
        "joint_pos": torch.zeros(29),
        "joint_vel": torch.zeros(29),
    }
    values.update(overrides)
    return ReferenceFrame(**values)


def _wrench(
    target: ReferenceFrame, state: RobotState | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return DirectController(DirectConfig())._root_wrench(target, state or _state())


def test_root_wrench_is_zero_for_matching_reference() -> None:
    force, torque = _wrench(_reference())

    torch.testing.assert_close(force, torch.zeros(3))
    torch.testing.assert_close(torque, torch.zeros(3))


def test_root_wrench_pulls_toward_positive_position_error() -> None:
    target = _reference(
        root_pos_w=_state().root_pos_w + torch.tensor([1.0, 0.0, 0.0])
    )
    force, _ = _wrench(target)

    assert force[0] > 0.0


def test_root_wrench_damps_excess_positive_velocity() -> None:
    state = _state()
    state = RobotState(
        **{**state.__dict__, "root_lin_vel_w": torch.tensor([1.0, 0.0, 0.0])}
    )
    force, _ = _wrench(_reference(), state)

    assert force[0] < 0.0


def test_root_wrench_applies_positive_torque_for_positive_yaw_error() -> None:
    yaw = torch.tensor(0.2)
    target_quat = torch.stack(
        (torch.cos(yaw / 2), torch.tensor(0.0), torch.tensor(0.0), torch.sin(yaw / 2))
    )
    _, torque = _wrench(_reference(root_quat_w=target_quat))

    assert torque[2] > 0.0


def test_root_wrench_clamps_force_and_torque_by_vector_norm() -> None:
    config = DirectConfig(max_force=10.0, max_torque=5.0)
    controller = DirectController(config)
    target = _reference(
        root_pos_w=_state().root_pos_w + torch.tensor([100.0, 100.0, 100.0]),
        root_quat_w=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    force, torque = controller._root_wrench(target, _state())

    torch.testing.assert_close(torch.linalg.vector_norm(force), torch.tensor(10.0))
    torch.testing.assert_close(torch.linalg.vector_norm(torque), torch.tensor(5.0))


def test_direct_controller_outputs_joint_targets_and_pelvis_wrench() -> None:
    controller = DirectController(DirectConfig())
    controller.load_motion(_motion(), _state())
    expected = controller.reference.current()

    output = controller.act(_state())

    torch.testing.assert_close(output.joint_target, expected.joint_pos)
    torch.testing.assert_close(output.joint_velocity_target, expected.joint_vel)
    assert len(output.external_wrenches) == 1
    assert output.external_wrenches[0].body == "pelvis"
