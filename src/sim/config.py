from __future__ import annotations

from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTUATOR_4010,
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_WAIST,
    get_g1_robot_cfg,
)
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from tasks import ObservationCameraSpec, TaskSpec

OBSERVATION_CAMERA = "observation_camera"


def make_sim_env_cfg(
    *,
    image_width: int = 640,
    image_height: int = 480,
    task: TaskSpec | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build the minimal 50 Hz MJLab environment for the simulated G1."""

    actuator_7520_14 = replace(
        G1_ACTUATOR_7520_14,
        target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
    )
    actuator_7520_22 = replace(
        G1_ACTUATOR_7520_22,
        target_names_expr=(
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_knee_joint",
        ),
    )
    actuators = (
        G1_ACTUATOR_5020,
        actuator_7520_14,
        actuator_7520_22,
        G1_ACTUATOR_4010,
        G1_ACTUATOR_WAIST,
        G1_ACTUATOR_ANKLE,
    )
    robot_cfg = get_g1_robot_cfg()
    robot_cfg.articulation = EntityArticulationInfoCfg(
        actuators=actuators,
        soft_joint_pos_limit_factor=0.9,
    )
    action_scale: dict[str, float] = {}
    for actuator in actuators:
        assert actuator.effort_limit is not None
        for pattern in actuator.target_names_expr:
            action_scale[pattern] = (
                0.25 * float(actuator.effort_limit) / actuator.stiffness
            )

    camera_spec = (
        task.observation_camera if task is not None else ObservationCameraSpec()
    )
    scene = SceneCfg(
        num_envs=1,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot_cfg},
        sensors=(
            CameraSensorCfg(
                name=OBSERVATION_CAMERA,
                width=image_width,
                height=image_height,
                data_types=("rgb", "depth"),
                fovy=camera_spec.fovy,
                clone_data=False,
            ),
        ),
    )
    if task is not None:
        scene.spec_fn = task.make_scene()

    return ManagerBasedRlEnvCfg(
        decimation=4,
        scene=scene,
        actions={
            "joint_position": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=action_scale,
                use_default_offset=True,
            )
        },
        sim=SimulationCfg(njmax=128, mujoco=MujocoCfg(timestep=0.005)),
        viewer=ViewerConfig(width=image_width, height=image_height, max_extra_envs=0),
        episode_length_s=0.0,
    )
