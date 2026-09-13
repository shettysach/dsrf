import numpy as np
import pytest
from tasks import get_task
from tasks.push_motion.settings import PushMotionSettings

from script.registry import create_task_script
from sim.root_path import RootPathController


def test_push_motion_box_starts_at_palm_reach_and_has_a_goal() -> None:
    task = get_task("push_motion")
    settings = PushMotionSettings()
    box = task.make_entities()["box"]

    assert task.virtual_force_objects == ("box",)
    assert box.init_state.pos == (settings.box_start_x, 0.0, 0.65)
    assert settings.box_start_x - settings.box_half_size[0] == pytest.approx(2.43)
    assert settings.box_goal_x - settings.box_start_x == pytest.approx(4.0)
    model = box.spec_fn().compile()
    assert model.body_mass[model.body("box").id] == settings.box_mass


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
    assert [target.frame for target in targets] == [51]
    assert targets[-1].root_xy == (0.8320000000000001, 0.0)


def test_root_path_uses_sparse_hand_keyframes() -> None:
    command = create_task_script("push_motion", "walk forward").next_command(0)
    assert command is not None and command.root_path_goal is not None
    state = np.array((2.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    controller = RootPathController(command.root_path_goal, window_seconds=2.08)
    controller.phase = "reach"

    reach_targets = controller.targets(state, frames=52, fps=25.0)
    assert [target.frame for target in reach_targets if target.end_effectors] == [
        51,
    ]
    assert [target.frame for target in reach_targets if target.root_xy is not None] == [
        51
    ]

    controller.phase = "push"
    push_targets = controller.targets(state, frames=52, fps=25.0)
    assert [target.frame for target in push_targets if target.end_effectors] == [
        51,
    ]
    assert [target.frame for target in push_targets if target.root_xy is not None] == [
        51
    ]

    reach = reach_targets[-1]
    push = push_targets[-1]
    assert all(target.palm_normal == (1.0, 0.0, 0.0) for target in reach.end_effectors)
    assert all(target.palm_normal == (1.0, 0.0, 0.0) for target in push.end_effectors)
    root_delta = np.array((*push.root_xy, 0.0))
    for reached, pushed in zip(reach.end_effectors, push.end_effectors, strict=True):
        np.testing.assert_allclose(
            np.asarray(pushed.target_xyz) - np.asarray(reached.target_xyz), root_delta
        )


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


def test_root_goal_keeps_its_absolute_deadline_across_measured_windows() -> None:
    command = create_task_script("push_motion", "walk forward").next_command(0)
    assert command is not None and command.root_path_goal is not None
    controller = RootPathController(command.root_path_goal, window_seconds=2.08)
    state = np.array((0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0))

    first = controller.targets(state, 52, 25.0, visible_frames=244)
    assert [sample.frame for sample in first] == [51, 124]
    assert first[-1].root_xy == (2.0, 0.0)
    controller.update(state, 2.08)  # The robot did not follow the generated path.
    second = controller.targets(state, 52, 25.0, visible_frames=244)
    assert [sample.frame for sample in second] == [51, 72]
    assert controller.phase == "approach"
    controller.update(state, 2.08)
    third = controller.targets(state, 52, 25.0, visible_frames=244)
    assert [sample.frame for sample in third] == [20, 51]
    assert third[0].root_xy == third[1].root_xy == (2.0, 0.0)
    controller.update(state, 2.08)
    controller.targets(state, 52, 25.0, visible_frames=244)
    assert controller.deadline == 280  # Explicitly replan only after missing frame 124.


def test_root_goal_requires_box_at_goal_and_measured_bilateral_reach() -> None:
    command = create_task_script("push_motion", "walk forward").next_command(0)
    assert command is not None and command.root_path_goal is not None
    state = np.array((6.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0))
    hands = {
        "left_hand": np.array((6.3, 0.2, 1.0)),
        "right_hand": np.array((6.3, -0.2, 1.0)),
    }
    controller = RootPathController(command.root_path_goal, window_seconds=2.08)
    controller.phase = "push"
    controller.update(state, 2.08, hands, np.array((6.0, 0.0, 0.65)))
    assert controller.phase == "push"
    controller.update(state, 2.08, hands, np.array((6.93, 0.0, 0.65)))
    assert controller.phase == "done"

    controller = RootPathController(command.root_path_goal, window_seconds=2.08)
    controller.phase = "push"
    controller.update(
        state,
        2.08,
        {"left_hand": hands["left_hand"], "right_hand": np.array((6.1, 0.0, 1.0))},
        np.array((6.93, 0.0, 0.65)),
    )
    assert controller.phase == "push"
