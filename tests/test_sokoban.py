import mujoco
import numpy as np
import pytest
from tasks import TASKS, get_task
from tasks.sokoban.scene import (
    BOX_MASS,
    BOX_STARTS,
    GOAL_CENTERS,
    MJ_JOINT_SLIDE,
    make_sokoban_spec_fn,
)

from sim.config import make_sim_env_cfg


def test_catalog_contains_sokoban() -> None:
    task = get_task("sokoban")

    assert task is TASKS["sokoban"]
    assert task.objective == "Push both boxes onto the two marked goal regions."
    assert task.observation_camera.distance == 5.0
    assert task.observation_camera.elevation == -50.0


def test_sokoban_uses_elevated_observation_framing() -> None:
    cfg = make_sim_env_cfg(task=get_task("sokoban"))

    assert cfg.scene.spec_fn is not None
    assert cfg.scene.sensors[0].data_types == ("rgb", "depth")


def test_sokoban_scene_has_two_pushable_boxes_and_two_goals() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_sokoban_spec_fn()(spec)
    model = spec.compile()

    box_bodies = [model.body(name) for name in ("sokoban_box_1", "sokoban_box_2")]
    goal_geoms = [model.geom(name) for name in ("sokoban_goal_1", "sokoban_goal_2")]

    assert len(BOX_STARTS) == 2
    assert len(GOAL_CENTERS) == 2
    assert [model.body_mass[body.id] for body in box_bodies] == pytest.approx(
        [BOX_MASS, BOX_MASS]
    )
    assert all(model.geom_contype[goal.id] == 0 for goal in goal_geoms)
    assert all(model.geom_conaffinity[goal.id] == 0 for goal in goal_geoms)
    assert model.ncam == 0

    for box_index in (1, 2):
        x_joint = model.joint(f"sokoban_box_{box_index}_x")
        y_joint = model.joint(f"sokoban_box_{box_index}_y")
        assert model.jnt_type[x_joint.id] == MJ_JOINT_SLIDE
        assert model.jnt_type[y_joint.id] == MJ_JOINT_SLIDE
        np.testing.assert_array_equal(model.jnt_axis[x_joint.id], (1.0, 0.0, 0.0))
        np.testing.assert_array_equal(model.jnt_axis[y_joint.id], (0.0, 1.0, 0.0))


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
