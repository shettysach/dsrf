import mujoco
import pytest
from tasks import TASKS, get_task
from tasks.endtest.scene import (
    TARGET_CENTER_X,
    TARGET_CENTER_Y,
    TARGET_HALF_SIZE,
    make_endtest_spec_fn,
)


def test_catalog_contains_endtest() -> None:
    task = get_task("endtest")

    assert task is TASKS["endtest"]
    assert task.objective == "Place each foot on its matching green target square."


def test_endtest_has_two_non_colliding_green_foot_targets() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_endtest_spec_fn()(spec)
    model = spec.compile()

    left = model.geom("endtest_left_target_surface")
    right = model.geom("endtest_right_target_surface")

    assert tuple(model.body_pos[model.body("endtest_left_target").id]) == pytest.approx(
        (TARGET_CENTER_X, TARGET_CENTER_Y, TARGET_HALF_SIZE[2])
    )
    assert tuple(model.body_pos[model.body("endtest_right_target").id]) == pytest.approx(
        (TARGET_CENTER_X, -TARGET_CENTER_Y, TARGET_HALF_SIZE[2])
    )
    assert tuple(model.geom_size[left.id]) == pytest.approx(TARGET_HALF_SIZE)
    assert tuple(model.geom_rgba[left.id]) == pytest.approx((0.1, 0.85, 0.25, 1.0))
    assert model.geom_contype[left.id] == 0
    assert model.geom_conaffinity[right.id] == 0
