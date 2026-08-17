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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from tasks import TaskSpec, ViewerSpec


def make_sim_env_cfg(
    *,
    image_width: int = 640,
    image_height: int = 480,
    task: TaskSpec | None = None,
    goal_index: int | None = None,
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

    scene = SceneCfg(
        num_envs=1,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot_cfg},
    )
    if task is not None:
        scene.spec_fn = (
            task.make_scene_with_goal(goal_index)
            if task.make_scene_with_goal is not None
            else task.make_scene()
        )

    viewer_spec = task.viewer if task is not None else ViewerSpec()

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
        viewer=ViewerConfig(
            origin_type=(
                ViewerConfig.OriginType.WORLD
                if viewer_spec.origin == "world"
                else ViewerConfig.OriginType.ASSET_BODY
            ),
            entity_name="robot" if viewer_spec.origin == "robot" else None,
            body_name="torso_link" if viewer_spec.origin == "robot" else None,
            lookat=viewer_spec.lookat,
            distance=viewer_spec.distance,
            elevation=viewer_spec.elevation,
            azimuth=viewer_spec.azimuth,
            width=image_width,
            height=image_height,
            max_extra_envs=0,
        ),
        episode_length_s=0.0,
    )
