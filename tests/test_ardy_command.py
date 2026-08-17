import math
from types import SimpleNamespace

import pytest
import torch
from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
from ardy.skeleton import G1Skeleton34

from motion_gen.ardy.constraints import build_constraints
from shared.messages import EndEffectorTarget


def _conditions():
    received: dict[str, object] = {}

    def create_conditions(index, data, length, normalize, device):
        received.update(
            index=index,
            data=data,
            length=length,
            normalize=normalize,
            device=device,
        )
        return torch.zeros((129, 8)), torch.zeros((129, 8))

    skeleton = SimpleNamespace(
        root_idx=0,
        bone_order_names=[
            "pelvis_skel",
            "left_hand_roll_skel",
            "right_hand_roll_skel",
            "left_toe_base",
            "right_toe_base",
        ],
    )
    return SimpleNamespace(create_conditions=create_conditions, skeleton=skeleton), received


def test_local_waypoint_becomes_ardy_root_endpoint() -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])

    motion_mask, observed_motion = build_constraints(
        motion_rep,
        root_history,
        torch.tensor(0.0),
        ((0.8, 0.3),),
        (),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert motion_mask.shape == (1, 129, 8)
    assert observed_motion.shape == (1, 129, 8)
    assert received["index"]["root_2d"][0].tolist() == [128]
    torch.testing.assert_close(
        received["data"]["root_2d"][0],
        torch.tensor([[1.3, 2.8]]),
    )


@pytest.mark.parametrize(
    ("heading", "expected_endpoint"),
    [
        (0.0, [1.0, 3.0]),
        (math.pi / 2.0, [0.0, 2.0]),
        (math.pi, [1.0, 1.0]),
        (-math.pi / 2.0, [2.0, 2.0]),
    ],
)
def test_local_forward_rotates_by_ardy_root_heading(
    heading: float, expected_endpoint: list[float]
) -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])

    build_constraints(
        motion_rep,
        root_history,
        torch.tensor(heading),
        ((1.0, 0.0),),
        (),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        received["data"]["root_2d"][0],
        torch.tensor([expected_endpoint]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_multiple_waypoints_are_evenly_spaced_through_the_generation() -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])

    build_constraints(
        motion_rep,
        root_history,
        torch.tensor(0.0),
        ((1.0, 0.0), (0.0, 1.0)),
        (),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert received["index"]["root_2d"][0].tolist() == [65, 128]
    torch.testing.assert_close(
        received["data"]["root_2d"][0],
        torch.tensor([[1.0, 3.0], [2.0, 2.0]]),
    )


def test_end_effector_becomes_final_global_joint_position() -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.8, 2.0], [1.0, 0.8, 2.0]])

    build_constraints(
        motion_rep,
        root_history,
        torch.tensor(0.0),
        (),
        (EndEffectorTarget("right_hand", (0.4, 0.2, 0.3)),),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert received["index"]["root_2d"][0].tolist() == [128]
    assert received["index"]["root_y_pos"][0].tolist() == [128]
    assert received["index"]["global_joints_positions"][0].tolist() == [
        [128, 0],
        [128, 2],
    ]
    torch.testing.assert_close(
        received["data"]["global_joints_positions"][0],
        torch.tensor([[1.0, 0.8, 2.0], [1.2, 1.1, 2.4]]),
    )


def test_waypoint_and_end_effector_share_the_final_constraint_frame() -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.8, 2.0], [1.0, 0.8, 2.0]])

    build_constraints(
        motion_rep,
        root_history,
        torch.tensor(0.0),
        ((1.0, 0.0),),
        (EndEffectorTarget("right_hand", (0.4, 0.2, 0.3)),),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert received["index"]["root_2d"][0].tolist() == [128]
    assert received["index"]["global_joints_positions"][0].tolist() == [
        [128, 0],
        [128, 2],
    ]
    torch.testing.assert_close(
        received["data"]["root_2d"][0], torch.tensor([[1.0, 3.0]])
    )
    torch.testing.assert_close(
        received["data"]["global_joints_positions"][0],
        torch.tensor([[1.0, 0.8, 3.0], [1.2, 1.1, 2.4]]),
    )


def test_foot_becomes_final_toe_position() -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.8, 2.0], [1.0, 0.8, 2.0]])

    build_constraints(
        motion_rep,
        root_history,
        torch.tensor(0.0),
        (),
        (EndEffectorTarget("left_foot", (0.4, 0.2, -0.8)),),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert received["index"]["global_joints_positions"][0].tolist() == [
        [128, 0],
        [128, 3],
    ]
    torch.testing.assert_close(
        received["data"]["global_joints_positions"][0],
        torch.tensor([[1.0, 0.8, 2.0], [1.2, 0.0, 2.4]]),
    )


def test_end_effector_compiles_with_ardy_motion_representation() -> None:
    motion_rep = ArdyMotionRep(G1Skeleton34(), 25)
    motion_rep.stats = SimpleNamespace(normalize=lambda value: value)

    motion_mask, observed_motion = build_constraints(
        motion_rep,
        torch.tensor([[0.0, 0.8, 0.0], [0.0, 0.8, 0.0]]),
        torch.tensor(0.0),
        (),
        (EndEffectorTarget("right_hand", (0.4, -0.2, 0.2)),),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert motion_mask.shape == observed_motion.shape == (1, 129, 414)
    assert int(motion_mask.sum()) == 6
    assert torch.isfinite(observed_motion).all()
