import mujoco
import numpy as np
import pytest
from tasks import TASKS, get_task
from tasks.sokoban.scene import (
    ARENA_HALF_WIDTH,
    ARENA_MAX_X,
    BOX_MASS,
    GOAL_HALF_SIZE,
    MJ_JOINT_SLIDE,
    SOKOBAN_VARIANTS,
    WALL_HALF_THICKNESS,
    make_sokoban_spec_fn,
)

from sim.config import make_sim_env_cfg


def test_catalog_contains_sokoban() -> None:
    task = get_task("sokoban")

    assert task is TASKS["sokoban"]
    assert task.objective == "Push the yellow box onto the marked green goal region."
    assert task.viewer.distance == 6.5
    assert task.viewer.elevation == -50.0


def test_sokoban_uses_elevated_viewer_framing() -> None:
    cfg = make_sim_env_cfg(task=get_task("sokoban"), goal_index=1)

    assert cfg.scene.spec_fn is not None
    assert cfg.viewer.distance == 6.5
    assert cfg.viewer.elevation == -50.0


@pytest.mark.parametrize("variant_index", range(3))
def test_sokoban_scene_has_one_pushable_box_and_one_edge_goal(
    variant_index: int,
) -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_sokoban_spec_fn(variant_index=variant_index)(spec)
    model = spec.compile()

    box_body = model.body("sokoban_box_1")
    box_geom = model.geom("sokoban_box_1_pushable")
    goal_geom = model.geom("sokoban_goal_1")

    assert ARENA_HALF_WIDTH == 3.5
    box_start, goal_center = SOKOBAN_VARIANTS[variant_index]
    assert tuple(model.body_pos[box_body.id, :2]) == pytest.approx(box_start)
    assert tuple(model.geom_pos[goal_geom.id, :2]) == pytest.approx(goal_center)
    assert (
        goal_center[0] + GOAL_HALF_SIZE == pytest.approx(
            ARENA_MAX_X - WALL_HALF_THICKNESS
        )
        or abs(goal_center[1]) + GOAL_HALF_SIZE
        == pytest.approx(ARENA_HALF_WIDTH - WALL_HALF_THICKNESS)
    )
    assert model.body_mass[box_body.id] == pytest.approx(BOX_MASS)
    assert not box_geom.name.endswith("_collision")
    assert model.geom_contype[goal_geom.id] == 0
    assert model.geom_conaffinity[goal_geom.id] == 0
    assert model.ncam == 0

    x_joint = model.joint("sokoban_box_1_x")
    y_joint = model.joint("sokoban_box_1_y")
    assert model.jnt_type[x_joint.id] == MJ_JOINT_SLIDE
    assert model.jnt_type[y_joint.id] == MJ_JOINT_SLIDE
    np.testing.assert_array_equal(model.jnt_axis[x_joint.id], (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(model.jnt_axis[y_joint.id], (0.0, 1.0, 0.0))


def test_sokoban_rejects_unknown_variation() -> None:
    with pytest.raises(ValueError, match="variant_index"):
        make_sokoban_spec_fn(variant_index=len(SOKOBAN_VARIANTS))


def test_sokoban_box_moves_under_a_small_planar_force() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_sokoban_spec_fn()(spec)
    model = spec.compile()
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    joint = model.joint("sokoban_box_1_x")
    qpos_address = model.jnt_qposadr[joint.id]
    dof_address = model.jnt_dofadr[joint.id]

    data.qfrc_applied[dof_address] = 1.0
    for _ in range(20):
        mujoco.mj_step(model, data)  # ty: ignore[unresolved-attribute]

    assert data.qpos[qpos_address] > 0.001
