import math
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from ardy.constraints import RightHandConstraintSet, Root2DConstraintSet
from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep
from ardy.skeleton import G1Skeleton34

from motion_gen.ardy.constraints import build_constraints
from shared.messages import EndEffectorTarget


def _conditions():
    received: dict[str, object] = {}
    names = [
        "pelvis_skel",
        "left_hip_pitch_skel",
        "right_hip_pitch_skel",
        "left_wrist_yaw_skel",
        "left_hand_roll_skel",
        "right_wrist_yaw_skel",
        "right_hand_roll_skel",
        "left_toe_base",
        "right_toe_base",
    ]
    skeleton = SimpleNamespace(
        root_idx=0,
        nbjoints=len(names),
        bone_order_names=names,
        bone_index={name: index for index, name in enumerate(names)},
        hip_joint_idx=[2, 1],
    )

    def expand_joint_names(values):
        positions = {
            "LeftHand": names[3:5],
            "RightHand": names[5:7],
            "LeftFoot": [names[7]],
            "RightFoot": [names[8]],
            "Hips": [names[0]],
        }
        pos_names = [name for value in values for name in positions[value]]
        rot_names = [positions[value][0] for value in values]
        return rot_names, pos_names

    skeleton.expand_joint_names = expand_joint_names

    def create_conditions(constraints, length, normalize, device):
        index = defaultdict(list)
        data = defaultdict(list)
        for constraint in constraints:
            constraint.update_constraints(data, index)
        received.update(
            constraints=constraints,
            index=index,
            data=data,
            length=length,
            normalize=normalize,
            device=device,
        )
        return torch.zeros((length, 64)), torch.zeros((length, 64))

    return (
        SimpleNamespace(
            create_conditions_from_constraints=create_conditions,
            normalize=lambda value: value,
            skeleton=skeleton,
        ),
        received,
    )


def _reference(motion_rep, root=(1.0, 0.8, 2.0)):
    joints = motion_rep.skeleton.nbjoints
    positions = torch.zeros((1, 1, joints, 3))
    positions[0, 0, 0] = torch.tensor(root)
    positions[0, 0, 1] = torch.tensor([0.2, 0.8, 2.0])
    positions[0, 0, 2] = torch.tensor([-0.2, 0.8, 2.0])
    positions[0, 0, 3] = torch.tensor([0.8, 1.0, 2.1])
    positions[0, 0, 4] = torch.tensor([0.8, 1.0, 2.2])
    positions[0, 0, 5] = torch.tensor([1.1, 1.0, 2.1])
    positions[0, 0, 6] = torch.tensor([1.1, 1.0, 2.2])
    rotations = torch.eye(3).expand(1, 1, joints, 3, 3).clone()
    return {"posed_joints": positions, "global_rot_mats": rotations}


def test_local_waypoint_becomes_ardy_root_endpoint() -> None:
    motion_rep, received = _conditions()
    build_constraints(
        motion_rep,
        torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        torch.tensor(0.0),
        ((0.8, 0.3),),
        (),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )
    assert isinstance(received["constraints"][0], Root2DConstraintSet)
    assert received["index"]["root_2d"][0].tolist() == [128]
    torch.testing.assert_close(
        received["data"]["root_2d"][0], torch.tensor([[1.3, 2.8]])
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
    build_constraints(
        motion_rep,
        torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
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


def test_multiple_waypoints_are_evenly_spaced() -> None:
    motion_rep, received = _conditions()
    build_constraints(
        motion_rep,
        torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        torch.tensor(0.0),
        ((1.0, 0.0), (0.0, 1.0)),
        (),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )
    assert received["index"]["root_2d"][0].tolist() == [65, 128]


def test_end_effector_requires_a_reference_pose() -> None:
    motion_rep, _ = _conditions()
    with pytest.raises(ValueError, match="reference pose"):
        build_constraints(
            motion_rep,
            torch.tensor([[1.0, 0.8, 2.0], [1.0, 0.8, 2.0]]),
            torch.tensor(0.0),
            (),
            (EndEffectorTarget("right_hand", (0.4, 0.2, 0.3)),),
            generated_frames=125,
            history_frames=4,
            device=torch.device("cpu"),
        )


def test_native_ee_translates_wrist_and_hand_and_preserves_root() -> None:
    motion_rep, received = _conditions()
    reference = _reference(motion_rep)
    build_constraints(
        motion_rep,
        torch.tensor([[1.0, 0.8, 2.0], [1.0, 0.8, 2.0]]),
        torch.tensor(0.0),
        (),
        (EndEffectorTarget("right_hand", (0.4, 0.2, 0.3)),),
        reference,
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert isinstance(received["constraints"][0], RightHandConstraintSet)
    assert received["index"]["global_joints_positions"][0].tolist() == [
        [128, 5],
        [128, 6],
        [128, 0],
    ]
    torch.testing.assert_close(
        received["data"]["global_joints_positions"][0],
        torch.tensor(
            [[1.2, 1.1, 2.3], [1.2, 1.1, 2.4], [1.0, 0.8, 2.0]]
        ),
    )
    torch.testing.assert_close(
        received["data"]["global_joints_rots"][0],
        torch.eye(3).expand(2, 3, 3),
    )
    assert received["index"]["root_y_pos"][0].tolist() == [128]
    assert received["index"]["global_root_heading"][0].tolist() == [128]


def test_waypoint_and_native_ee_share_final_frame() -> None:
    motion_rep, received = _conditions()
    reference = _reference(motion_rep, root=(1.0, 0.8, 3.0))
    build_constraints(
        motion_rep,
        torch.tensor([[1.0, 0.8, 2.0], [1.0, 0.8, 2.0]]),
        torch.tensor(0.0),
        ((1.0, 0.0),),
        (EndEffectorTarget("right_hand", (0.4, 0.2, 0.3)),),
        reference,
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )
    assert [value.tolist() for value in received["index"]["root_2d"]] == [
        [128],
        [128],
    ]


def test_native_ee_compiles_with_ardy_motion_representation() -> None:
    motion_rep = ArdyMotionRep(G1Skeleton34(), 25)
    motion_rep.stats = SimpleNamespace(normalize=lambda value: value)
    joints = motion_rep.skeleton.nbjoints
    reference = {
        "posed_joints": torch.zeros((1, 1, joints, 3)),
        "global_rot_mats": torch.eye(3).expand(1, 1, joints, 3, 3).clone(),
    }
    reference["posed_joints"][0, 0, motion_rep.skeleton.root_idx, 1] = 0.8
    right_hip, left_hip = motion_rep.skeleton.hip_joint_idx
    reference["posed_joints"][0, 0, right_hip, 0] = -0.2
    reference["posed_joints"][0, 0, left_hip, 0] = 0.2

    motion_mask, observed_motion = build_constraints(
        motion_rep,
        torch.tensor([[0.0, 0.8, 0.0], [0.0, 0.8, 0.0]]),
        torch.tensor(0.0),
        (),
        (EndEffectorTarget("right_hand", (0.4, -0.2, 0.2)),),
        reference,
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )
    assert motion_mask.shape == observed_motion.shape == (1, 129, 414)
    assert int(motion_mask.sum()) == 23
    assert torch.isfinite(observed_motion).all()
