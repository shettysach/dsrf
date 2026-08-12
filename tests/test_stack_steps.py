import mujoco
import pytest
from tasks import TASKS, get_task
from tasks.stack_steps.scene import (
    ARENA_HALF_WIDTH,
    BOTTOM_STEP_HALF_SIZE,
    BOTTOM_STEP_MASS,
    PLATFORM_BACK_X,
    PLATFORM_FRONT_X,
    PLATFORM_HALF_WIDTH,
    PLATFORM_HEIGHT,
    TOP_STEP_HALF_SIZE,
    TOP_STEP_MASS,
    make_stack_steps_spec_fn,
)

from sim.config import make_sim_env_cfg


def test_catalog_contains_stack_steps() -> None:
    task = get_task("stack-steps")

    assert task is TASKS["stack-steps"]
    assert task.objective == (
        "Stack the two blocks into steps and reach the green platform."
    )
    assert task.viewer.distance == 4.0
    assert task.viewer.elevation == -28.0


def test_stack_steps_config_applies_scene_and_viewer() -> None:
    cfg = make_sim_env_cfg(task=get_task("stack-steps"))

    assert cfg.scene.spec_fn is not None
    assert cfg.viewer.distance == 4.0
    assert cfg.viewer.elevation == -28.0


def test_stack_steps_scene_has_walls_goal_platform_and_two_blocks() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_stack_steps_spec_fn(seed=1234)(spec)
    model = spec.compile()

    wall_bodies = [body for body in spec.bodies if body.name.endswith("_wall")]
    platform = model.body("stack_steps_platform")
    platform_base = model.geom("stack_steps_platform_base")
    platform_goal = model.geom("stack_steps_platform_goal")
    bottom = model.body("stack_steps_bottom_step")
    bottom_geom = model.geom("stack_steps_bottom_step_pushable")
    top = model.body("stack_steps_top_step")
    top_geom = model.geom("stack_steps_top_step_pushable")

    assert len(wall_bodies) == 4
    assert ARENA_HALF_WIDTH == 2.5
    assert model.body_pos[platform.id, 2] + model.geom_pos[
        platform_goal.id, 2
    ] + model.geom_size[
        platform_goal.id, 2
    ] == pytest.approx(PLATFORM_HEIGHT)
    assert model.body_pos[platform.id, 0] - model.geom_size[
        platform_base.id, 0
    ] == pytest.approx(PLATFORM_BACK_X)
    assert model.body_pos[platform.id, 0] + model.geom_size[
        platform_base.id, 0
    ] == pytest.approx(PLATFORM_FRONT_X)
    assert PLATFORM_HALF_WIDTH < ARENA_HALF_WIDTH - 0.5
    assert tuple(model.geom_rgba[platform_base.id]) == pytest.approx(
        (0.3, 0.35, 0.4, 1.0)
    )
    assert tuple(model.geom_rgba[platform_goal.id]) == pytest.approx(
        (0.15, 0.8, 0.3, 1.0)
    )
    assert tuple(model.geom_size[bottom_geom.id]) == pytest.approx(
        BOTTOM_STEP_HALF_SIZE
    )
    assert tuple(model.geom_size[top_geom.id]) == pytest.approx(TOP_STEP_HALF_SIZE)
    assert model.body_mass[bottom.id] == pytest.approx(BOTTOM_STEP_MASS)
    assert model.body_mass[top.id] == pytest.approx(TOP_STEP_MASS)
    assert BOTTOM_STEP_HALF_SIZE[0] > TOP_STEP_HALF_SIZE[0]
    assert BOTTOM_STEP_HALF_SIZE[1] > TOP_STEP_HALF_SIZE[1]
    assert PLATFORM_HEIGHT == pytest.approx(
        3.0 * BOTTOM_STEP_HALF_SIZE[2] * 2.0
    )
    assert model.ncam == 0

    assert model.jnt_type[model.joint("stack_steps_bottom_step_free").id] == (
        mujoco.mjtJoint.mjJNT_FREE  # ty: ignore[unresolved-attribute]
    )
    assert model.jnt_type[model.joint("stack_steps_top_step_free").id] == (
        mujoco.mjtJoint.mjJNT_FREE  # ty: ignore[unresolved-attribute]
    )


def test_stack_steps_seed_randomizes_non_overlapping_block_positions() -> None:
    def positions(seed: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
        make_stack_steps_spec_fn(seed=seed)(spec)
        model = spec.compile()
        return (
            tuple(model.body("stack_steps_bottom_step").pos[:2]),
            tuple(model.body("stack_steps_top_step").pos[:2]),
        )

    first = positions(1234)
    repeated = positions(1234)
    different = positions(5678)

    assert first == repeated
    assert first != different
    assert (
        abs(first[0][0] - first[1][0])
        >= BOTTOM_STEP_HALF_SIZE[0] + TOP_STEP_HALF_SIZE[0]
        or abs(first[0][1] - first[1][1])
        >= BOTTOM_STEP_HALF_SIZE[1] + TOP_STEP_HALF_SIZE[1]
    )


def test_stack_steps_blocks_settle_under_gravity() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    spec.worldbody.add_geom(
        name="test_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,  # ty: ignore[unresolved-attribute]
        size=(10.0, 10.0, 0.1),
    )
    make_stack_steps_spec_fn(seed=1234)(spec)
    model = spec.compile()
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]

    for _ in range(100):
        mujoco.mj_step(model, data)  # ty: ignore[unresolved-attribute]

    for name, half_size in (
        ("stack_steps_bottom_step_free", BOTTOM_STEP_HALF_SIZE),
        ("stack_steps_top_step_free", TOP_STEP_HALF_SIZE),
    ):
        joint = model.joint(name)
        qpos_address = model.jnt_qposadr[joint.id]
        assert data.qpos[qpos_address + 2] == pytest.approx(half_size[2], abs=1e-3)
