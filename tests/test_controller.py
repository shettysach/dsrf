from types import SimpleNamespace

import numpy as np
import pytest
import torch

from controller import ControlOutput, ExternalWrench, RobotState, RootTarget
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pos_w", torch.zeros(2)),
        ("quat_w", torch.zeros(3)),
        ("lin_vel_w", torch.zeros(2)),
        ("ang_vel_w", torch.zeros(2)),
    ),
)
def test_root_target_rejects_bad_shapes(field: str, value: torch.Tensor) -> None:
    values = {
        "pos_w": torch.zeros(3),
        "quat_w": torch.zeros(4),
        "lin_vel_w": torch.zeros(3),
        "ang_vel_w": torch.zeros(3),
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        RootTarget(**values)


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


def test_direct_controller_tracks_joint_reference() -> None:
    controller = DirectController(DirectConfig(pin_root=False))
    controller.load_motion(_motion(), _state())

    first = controller.act(_state())
    second = controller.act(_state())

    torch.testing.assert_close(first.joint_target, torch.zeros(29))
    torch.testing.assert_close(second.joint_target, torch.arange(29).float())
    torch.testing.assert_close(
        first.joint_velocity_target, torch.arange(29).float() * 50
    )
    torch.testing.assert_close(
        second.joint_velocity_target, torch.arange(29).float() * 50
    )
    assert first.external_wrenches == ()
    assert first.root_target is None
    assert not first.completed
    assert second.completed


def test_direct_controller_includes_reference_root_when_pinning_enabled() -> None:
    controller = DirectController(DirectConfig(pin_root=True))
    controller.load_motion(_motion(), _state())
    expected = controller.reference.current()

    output = controller.act(_state())

    assert output.root_target is not None
    torch.testing.assert_close(output.root_target.pos_w, expected.root_pos_w)
    torch.testing.assert_close(output.root_target.quat_w, expected.root_quat_w)
    torch.testing.assert_close(output.root_target.lin_vel_w, expected.root_lin_vel_w)
    torch.testing.assert_close(output.root_target.ang_vel_w, expected.root_ang_vel_w)
    assert output.external_wrenches == ()
