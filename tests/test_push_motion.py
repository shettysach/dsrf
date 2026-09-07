import numpy as np
from tasks import get_task

from script.tasks.push_motion import PushMotionScript
from sim.push_motion import PushMotionController, PushMotionState


def test_push_motion_task_has_no_entities_or_physical_assistance() -> None:
    task = get_task("push_motion")

    assert task.make_entities() == {}
    assert task.virtual_force_objects == ()


def _state(x: float) -> PushMotionState:
    return PushMotionState(
        np.array((x, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    )


def test_push_motion_advances_without_any_contact_state() -> None:
    command = PushMotionScript("execute the bilateral push motion").next_command(0)
    assert command is not None and command.push_motion_goal is not None
    controller = PushMotionController(
        command.push_motion_goal, _state(0.0), window_seconds=2.08
    )

    controller.update(_state(2.0), 0.02)
    assert controller.phase == "reach"
    controller.update(_state(2.0), 2.08)
    assert controller.phase == "push"
    controller.update(_state(6.0), 0.02)

    assert controller.phase == "done"


def test_push_motion_carries_palm_targets_with_the_base() -> None:
    command = PushMotionScript("execute the bilateral push motion").next_command(0)
    assert command is not None and command.push_motion_goal is not None
    controller = PushMotionController(
        command.push_motion_goal, _state(2.0), window_seconds=2.08
    )
    controller.phase = "push"

    samples = controller.targets(_state(2.0), frames=52, fps=25.0)
    left = samples[-1].end_effectors[0]

    assert samples[-1].root_xy == (0.312, 0.0)
    np.testing.assert_allclose(left.target_xyz, (0.892, 0.16, 0.40))
    assert left.palm_normal == (1.0, 0.0, 0.0)
    assert not samples[-1].root_upright
