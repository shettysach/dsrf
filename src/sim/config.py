from __future__ import annotations

import math
from dataclasses import replace

import torch
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
from mjlab.utils.lab_api.math import quat_from_matrix
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
    if task is not None and task.robot_initial_pos is not None:
        robot_cfg.init_state = replace(robot_cfg.init_state, pos=task.robot_initial_pos)
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
    camera_pos, camera_quat = _observation_camera_pose(camera_spec)
    entities = {"robot": robot_cfg}
    if task is not None:
        entities.update(task.make_entities())

    scene = SceneCfg(
        num_envs=1,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities=entities,
        sensors=(
            CameraSensorCfg(
                name=OBSERVATION_CAMERA,
                parent_body="robot/torso_link",
                pos=camera_pos,
                quat=camera_quat,
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

    actions = {
        "joint_position": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=action_scale,
            use_default_offset=True,
        )
    }
    return ManagerBasedRlEnvCfg(
        decimation=4,
        scene=scene,
        actions=actions,
        sim=SimulationCfg(njmax=128, mujoco=MujocoCfg(timestep=0.005)),
        viewer=ViewerConfig(width=image_width, height=image_height, max_extra_envs=0),
        episode_length_s=0.0,
    )


def _observation_camera_pose(
    spec: ObservationCameraSpec,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return a torso-relative observation-camera pose.

    Egocentric cameras are placed at the G1 head center and look along the
    robot's local +X axis.  The G1 model has no separate head body, so the
    camera remains attached to the torso while using the head's local offset.
    """
    if spec.egocentric:
        position = torch.tensor((0.0, 0.0, 0.43), dtype=torch.float64)
        right = position.new_tensor((0.0, -1.0, 0.0))
        up = position.new_tensor((0.0, 0.0, 1.0))
        back = position.new_tensor((-1.0, 0.0, 0.0))
        rotation = torch.stack((right, up, back), dim=1)
        quaternion = quat_from_matrix(rotation)
        return tuple(position.tolist()), tuple(quaternion.tolist())

    elevation = math.radians(spec.elevation)
    azimuth = math.radians(spec.azimuth)
    position = torch.tensor(
        (
            -spec.distance * math.cos(elevation) * math.cos(azimuth),
            -spec.distance * math.cos(elevation) * math.sin(azimuth),
            -spec.distance * math.sin(elevation),
        ),
        dtype=torch.float64,
    )
    forward = torch.nn.functional.normalize(-position, dim=0)
    world_up = position.new_tensor((0.0, 0.0, 1.0))
    right = torch.nn.functional.normalize(torch.cross(forward, world_up, dim=0), dim=0)
    up = torch.cross(right, forward, dim=0)
    rotation = torch.stack((right, up, -forward), dim=1)
    quaternion = quat_from_matrix(rotation)
    return tuple(position.tolist()), tuple(quaternion.tolist())
