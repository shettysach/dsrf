import numpy as np
import torch

from shared.messages import MotionChunk
from tracker import RobotState
from tracker.reference import MotionReference


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


def test_reference_starts_at_live_root_pose() -> None:
    reference = MotionReference("cpu")
    motion = _motion()
    motion.qpos[0, :7] = [1.0, 2.0, 1.3, 0.5, 0.5, 0.5, 0.5]
    state = _state()
    reference.load(motion, state.root_pos_w, state.root_quat_w)
    frame = reference.current()
    torch.testing.assert_close(frame.root_pos_w, state.root_pos_w)
    torch.testing.assert_close(frame.root_quat_w, state.root_quat_w)
