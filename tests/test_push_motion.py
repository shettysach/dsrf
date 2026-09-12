import numpy as np
from tasks import get_task
from tasks.push_motion.settings import PushMotionSettings

from script.registry import create_task_script
from sim.root_path import RootPathController


def test_push_motion_task_has_no_entities_or_physical_assistance() -> None:
    task = get_task("push_motion")

    assert task.make_entities() == {}
    assert task.virtual_force_objects == ()


def test_push_motion_uses_plain_text_generation_without_targets() -> None:
    script = create_task_script(
        "push_motion", "walk forward with both arms held forward"
    )

    command = script.next_command(0)

    assert command is not None
    assert command.target_xys == ()
    assert command.end_effectors == ()
    assert command.contact_goal is None
    assert command.root_path_goal is not None
    assert command.root_path_goal.approach_xy == (2.0, 0.0)
    assert command.root_path_goal.target_xy == (6.0, 0.0)
    left, right = command.root_path_goal.points
    assert left.palm_normal == right.palm_normal == (1.0, 0.0, 0.0)


def test_root_path_uses_only_timed_2d_root_targets_during_approach() -> None:
    command = create_task_script("push_motion", "walk forward").next_command(0)
    assert command is not None and command.root_path_goal is not None
    state = np.array((0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    controller = RootPathController(command.root_path_goal, window_seconds=2.08)

    targets = controller.targets(state, frames=52, fps=25.0)

    assert all(
        not target.end_effectors and not target.root_upright for target in targets
    )
    assert targets[-1].root_xy == (0.8320000000000001, 0.0)


def test_root_path_uses_sparse_hand_keyframes() -> None:
    command = create_task_script("push_motion", "walk forward").next_command(0)
    assert command is not None and command.root_path_goal is not None
    state = np.array((2.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    controller = RootPathController(command.root_path_goal, window_seconds=2.08)
    controller.phase = "reach"

    reach_targets = controller.targets(state, frames=52, fps=25.0)
    assert [target.frame for target in reach_targets if target.end_effectors] == [
        15,
        31,
        51,
    ]

    controller.phase = "push"
    push_targets = controller.targets(state, frames=52, fps=25.0)
    assert [target.frame for target in push_targets if target.end_effectors] == [
        9,
        19,
        29,
        39,
        51,
    ]


def test_root_path_can_skip_hand_targets_for_diagnosis() -> None:
    command = create_task_script("push_motion", "walk forward").next_command(0)
    assert command is not None and command.root_path_goal is not None
    controller = RootPathController(
        command.root_path_goal,
        window_seconds=2.08,
        config=PushMotionSettings(hand_targets=False),
    )
    controller.phase = "push"
    state = np.array((2.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)

    targets = controller.targets(state, frames=52, fps=25.0)

    assert all(not target.end_effectors for target in targets)
