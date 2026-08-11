import mujoco
import pytest
from tasks.catalog import TASKS, get_task
from tasks.portrait_corridors import make_portrait_corridors_spec_fn

from sim.config import make_sim_env_cfg


def test_catalog_contains_portrait_corridors() -> None:
    definition = get_task("portrait-corridors")

    assert definition is TASKS["portrait-corridors"]
    assert definition.objective == (
        "Stand in front of the image of the creator of Linux."
    )
    assert definition.camera_distance == 3.5
    assert definition.camera_elevation == -30.0


def test_catalog_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Available: portrait-corridors"):
        get_task("unknown")


def test_sonic_config_applies_task_scene_and_camera_distance() -> None:
    task_cfg = make_sim_env_cfg(task="portrait-corridors")
    plain_cfg = make_sim_env_cfg(task=None)

    assert task_cfg.scene.spec_fn is not None
    assert task_cfg.viewer.distance == 3.5
    assert task_cfg.viewer.elevation == -30.0
    assert plain_cfg.scene.spec_fn is None
    assert plain_cfg.viewer.distance == 2.0
    assert plain_cfg.viewer.elevation == -15.0


def test_sonic_config_uses_third_person_torso_camera() -> None:
    camera = make_sim_env_cfg().viewer

    assert camera.origin_type == camera.OriginType.ASSET_BODY
    assert camera.entity_name == "robot"
    assert camera.body_name == "torso_link"
    assert camera.azimuth == 0.0
    assert camera.elevation == -15.0


def test_portrait_corridors_spec_adds_portraits_walls_and_cameras() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_portrait_corridors_spec_fn()(spec)

    assert {body.name for body in spec.bodies if body.name.endswith("_portrait")} == {
        "portrait_corridors_linus_portrait",
        "portrait_corridors_karpathy_portrait",
        "portrait_corridors_nolan_portrait",
    }
    assert {texture.name for texture in spec.textures} == {
        "portrait_corridors_linus_texture",
        "portrait_corridors_karpathy_texture",
        "portrait_corridors_nolan_texture",
    }
    assert len([body for body in spec.bodies if body.name.endswith("_wall")]) == 5

    walls = {body.name: body for body in spec.bodies if body.name.endswith("_wall")}
    assert tuple(walls["portrait_corridors_end_wall"].geoms[0].rgba) == (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert "portrait_corridors_back_wall" not in walls

    cameras = {camera.name: camera for camera in spec.cameras}
    assert set(cameras) == {"corridor_left", "corridor_center", "corridor_right"}
    assert [tuple(camera.pos) for camera in cameras.values()] == pytest.approx(
        [(1.8, 2.0, 1.25), (1.8, 0.0, 1.25), (1.8, -2.0, 1.25)]
    )

    model = spec.compile()
    assert model.ntex == 3
    assert model.nmesh == 3
    assert model.ncam == 3
    assert model.mat_texid[:, 1].tolist() == [0, 1, 2]
    assert model.mesh_texcoordnum.tolist() == [4, 4, 4]


def test_portraits_fill_each_corridor() -> None:
    spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_portrait_corridors_spec_fn(seed=1234)(spec)

    positions = [
        tuple(body.pos) for body in spec.bodies if body.name.endswith("_portrait")
    ]
    assert len(positions) == 3
    assert sorted(position[1] for position in positions) == [-2.0, 0.0, 2.0]
    for x, _, z in positions:
        assert x == pytest.approx(5.89)
        assert z == pytest.approx(1.25)


def test_portrait_assignment_is_reproducible_with_a_seed() -> None:
    first_spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    second_spec = mujoco.MjSpec()  # ty: ignore[unresolved-attribute]
    make_portrait_corridors_spec_fn(seed=1234)(first_spec)
    make_portrait_corridors_spec_fn(seed=1234)(second_spec)

    def portrait_positions(spec) -> dict[str, tuple[float, ...]]:
        return {
            body.name: tuple(body.pos)
            for body in spec.bodies
            if body.name.endswith("_portrait")
        }

    assert portrait_positions(first_spec) == portrait_positions(second_spec)
