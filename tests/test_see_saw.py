import mujoco
import numpy as np
import pytest
from tasks import TASKS, get_task
from tasks.see_saw.scene import (
    COUNTERWEIGHT_LOCAL_POS,
    COUNTERWEIGHT_MASS,
    HINGE_DAMPING,
    HINGE_FRICTION_LOSS,
    INITIAL_TILT_DEGREES,
    MJ_JOINT_HINGE,
    PIVOT,
    PLANK_HALF_SIZE,
    hinge_limit_radians,
    make_see_saw_spec_fn,
)

from sim.config import make_sim_env_cfg


def test_catalog_contains_see_saw() -> None:
    task = get_task("see-saw")

    assert task is TASKS["see-saw"]
    assert task.objective == (
        "Walk onto the counterweighted see-saw and stand where it balances."
    )
    assert task.viewer.distance == 4.5
    assert task.viewer.elevation == -24.0


def test_see_saw_config_applies_scene_and_viewer() -> None:
    cfg = make_sim_env_cfg(task=get_task("see-saw"))

    assert cfg.scene.spec_fn is not None
    assert cfg.viewer.distance == 4.5
    assert cfg.viewer.elevation == -24.0


def test_see_saw_scene_has_walls_hinged_plank_and_counterweight() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_see_saw_spec_fn()(spec)
    model = spec.compile()

    wall_bodies = [body for body in spec.bodies if body.name.endswith("_wall")]
    plank = model.body("see_saw_plank")
    plank_geom = model.geom("see_saw_plank_surface")
    fulcrum = model.geom("see_saw_fulcrum_base")
    hinge = model.joint("see_saw_hinge")
    counterweight = model.geom("see_saw_counterweight")

    assert len(wall_bodies) == 4
    assert tuple(model.body_pos[plank.id]) == pytest.approx(PIVOT)
    assert tuple(model.geom_size[plank_geom.id]) == pytest.approx(PLANK_HALF_SIZE)
    assert model.geom_contype[fulcrum.id] == 0
    assert model.geom_conaffinity[fulcrum.id] == 0
    assert model.jnt_type[hinge.id] == MJ_JOINT_HINGE
    np.testing.assert_array_equal(model.jnt_axis[hinge.id], (1.0, 0.0, 0.0))
    assert tuple(model.jnt_range[hinge.id]) == pytest.approx(
        (-hinge_limit_radians(), hinge_limit_radians())
    )
    assert model.qpos0[model.jnt_qposadr[hinge.id]] == pytest.approx(
        np.deg2rad(INITIAL_TILT_DEGREES)
    )
    assert model.dof_damping[model.jnt_dofadr[hinge.id]] == pytest.approx(HINGE_DAMPING)
    assert model.dof_frictionloss[model.jnt_dofadr[hinge.id]] == pytest.approx(
        HINGE_FRICTION_LOSS
    )
    assert tuple(model.geom_pos[counterweight.id]) == pytest.approx(
        COUNTERWEIGHT_LOCAL_POS
    )
    assert model.body_mass[plank.id] > COUNTERWEIGHT_MASS
    assert "see_saw_balance_mark" not in {
        model.geom(index).name for index in range(model.ngeom)
    }
    assert model.ncam == 0


def test_counterweight_tilts_plank_to_limited_angle() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_see_saw_spec_fn()(spec)
    model = spec.compile()
    data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
    hinge = model.joint("see_saw_hinge")
    qpos_address = model.jnt_qposadr[hinge.id]

    for _ in range(2000):
        mujoco.mj_step(model, data)  # ty: ignore[unresolved-attribute]

    assert data.qpos[qpos_address] < 0.0
    # MuJoCo's soft limit permits a very small constraint penetration.
    assert data.qpos[qpos_address] >= -hinge_limit_radians() - 5e-4
