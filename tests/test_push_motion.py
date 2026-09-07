from tasks import get_task

from script.registry import create_task_script


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
    assert command.target_xys == ((2.0, 0.0), (6.0, 0.0))
    assert command.end_effectors == ()
    assert command.contact_goal is None
