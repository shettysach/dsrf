import mujoco
import numpy as np
import pytest
from tasks import TASKS, get_task
from tasks.sokoban.scene import (
    BOX_MASS,
    COMPLETED_BOX_RGBA,
    COMPLETED_BOX_CENTER_TOLERANCE,
    GRID_HEIGHT,
    GRID_WIDTH,
    MJ_JOINT_SLIDE,
    OUTER_WALL_HALF_THICKNESS,
    SokobanCompletionVisualizer,
    get_level,
    grid_to_world,
    level_positions,
    make_sokoban_spec_fn,
)

from sim.config import make_sim_env_cfg


def test_catalog_contains_sokoban() -> None:
    task = get_task("sokoban")

    assert task is TASKS["sokoban"]
    assert task.objective == "Push every yellow box onto a separate green goal region."
    assert task.robot_initial_pos == (*grid_to_world((4, 3)), 0.76)
    assert task.robot_initial_rot == pytest.approx(
        (0.7071067811865476, 0.0, 0.0, 0.7071067811865475)
    )
    assert task.observation_camera.world_position == (0.5, -4.5, 4.2)
    assert task.observation_camera.world_lookat == (0.5, 0.5, 0.0)
    assert task.observation_camera.follow_robot_translation


def test_sokoban_uses_elevated_observation_framing() -> None:
    cfg = make_sim_env_cfg(task=get_task("sokoban"))

    assert cfg.scene.spec_fn is not None
    assert cfg.scene.sensors[0].data_types == ("rgb", "depth")
    assert cfg.scene.sensors[0].parent_body is None
    assert cfg.scene.entities["robot"].init_state.rot == pytest.approx(
        get_task("sokoban").robot_initial_rot
    )


@pytest.mark.parametrize("level", range(1, 11))
def test_every_eval_preset_is_a_tile_for_tile_physical_scene(level: int) -> None:
    board = get_level(level)
    positions = level_positions(board)
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_sokoban_spec_fn(level)(spec)
    model = spec.compile()

    assert (GRID_WIDTH, GRID_HEIGHT) == (8, 8)
    assert len(positions.boxes) == len(positions.goals) in (2, 3)
    assert len(positions.walls) == sum(row.count("#") for row in board.rows)
    outer_wall = model.geom("sokoban_outer_north_wall_collision")
    assert model.geom_size[outer_wall.id, 1] == pytest.approx(OUTER_WALL_HALF_THICKNESS)
    assert model.geom_size[outer_wall.id, 0] == pytest.approx(3.0)
    outer_wall_body = model.body("sokoban_outer_north_wall")
    assert model.body_pos[outer_wall_body.id, 1] - model.geom_size[
        outer_wall.id, 1
    ] == pytest.approx(3.0)
    assert [
        model.body_mass[model.body(f"sokoban_box_{index}").id]
        for index in range(1, len(positions.boxes) + 1)
    ] == pytest.approx([BOX_MASS] * len(positions.boxes))
    for index, cell in enumerate(positions.boxes, 1):
        body = model.body(f"sokoban_box_{index}")
        np.testing.assert_array_equal(model.body_pos[body.id][:2], grid_to_world(cell))
        for axis in ("x", "y"):
            joint = model.joint(f"sokoban_box_{index}_{axis}")
            assert model.jnt_type[joint.id] == MJ_JOINT_SLIDE
    for index, cell in enumerate(positions.goals, 1):
        goal = model.geom(f"sokoban_goal_{index}")
        np.testing.assert_array_equal(model.geom_pos[goal.id][:2], grid_to_world(cell))
        assert model.geom_contype[goal.id] == 0
        assert model.geom_conaffinity[goal.id] == 0


def test_sokoban_box_moves_under_a_small_planar_force() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_sokoban_spec_fn(level=1)(spec)
    model = spec.compile()
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    joint = model.joint("sokoban_box_1_x")
    qpos_address = model.jnt_qposadr[joint.id]
    dof_address = model.jnt_dofadr[joint.id]

    data.qfrc_applied[dof_address] = 1.0
    for _ in range(20):
        mujoco.mj_step(model, data)  # ty: ignore[unresolved-attribute]

    assert data.qpos[qpos_address] > 0.001


def test_sokoban_box_is_dark_green_when_centred_on_a_goal() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_sokoban_spec_fn(level=1)(spec)
    model = spec.compile()
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    visualizer = SokobanCompletionVisualizer(model)
    box = model.geom("sokoban_box_1_collision")
    goal = model.geom("sokoban_goal_1")

    visualizer.update(data)
    assert tuple(model.geom_rgba[box.id]) != COMPLETED_BOX_RGBA

    for axis, coordinate in (("x", 0), ("y", 1)):
        joint = model.joint(f"sokoban_box_1_{axis}")
        data.qpos[model.jnt_qposadr[joint.id]] = (
            model.geom_pos[goal.id, coordinate]
            - model.body_pos[model.geom_bodyid[box.id], coordinate]
        )
    mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]
    assert not visualizer.update(data)

    np.testing.assert_allclose(model.geom_rgba[box.id], COMPLETED_BOX_RGBA)

    box_2 = model.geom("sokoban_box_2_collision")
    goal_2 = model.geom("sokoban_goal_2")
    for axis, coordinate in (("x", 0), ("y", 1)):
        joint = model.joint(f"sokoban_box_2_{axis}")
        data.qpos[model.jnt_qposadr[joint.id]] = (
            model.geom_pos[goal_2.id, coordinate]
            - model.body_pos[model.geom_bodyid[box_2.id], coordinate]
        )
    mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]

    assert visualizer.update(data)


def test_sokoban_box_is_complete_when_comfortably_centred_on_a_goal() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_sokoban_spec_fn(level=1)(spec)
    model = spec.compile()
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    visualizer = SokobanCompletionVisualizer(model)
    box = model.geom("sokoban_box_1_collision")
    goal = model.geom("sokoban_goal_1")

    for axis, coordinate in (("x", 0), ("y", 1)):
        joint = model.joint(f"sokoban_box_1_{axis}")
        offset = COMPLETED_BOX_CENTER_TOLERANCE * 0.9 if axis == "x" else 0.0
        data.qpos[model.jnt_qposadr[joint.id]] = (
            model.geom_pos[goal.id, coordinate]
            - model.body_pos[model.geom_bodyid[box.id], coordinate]
            + offset
        )
    mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]

    assert not visualizer.update(data)
    np.testing.assert_allclose(model.geom_rgba[box.id], COMPLETED_BOX_RGBA)
