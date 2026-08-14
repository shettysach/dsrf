from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from tasks import TaskSpec

from shared.geometry import yaw_from_quat_wxyz
from shared.messages import ProjectionContext
from sim.config import make_sim_env_cfg

if TYPE_CHECKING:
    from mjlab.envs.types import VecEnvObs, VecEnvStepReturn
    from mjlab.sim import Simulation


@dataclass(frozen=True)
class RobotState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class MjlabEnv:
    def __init__(
        self,
        *,
        device: str = "cpu",
        image_width: int = 640,
        image_height: int = 480,
        task: TaskSpec | None = None,
        goal_index: int | None = None,
        camera_yaw: bool = True,
    ) -> None:
        torch_device = torch.device(device)
        env_cfg = make_sim_env_cfg(
            image_width=image_width,
            image_height=image_height,
            task=task,
            goal_index=goal_index,
        )
        self._camera_base_azimuth = float(env_cfg.viewer.azimuth)
        self._camera_yaw = camera_yaw
        self._task = task
        self._env = ManagerBasedRlEnv(
            cfg=env_cfg,
            device=str(torch_device),
            render_mode="rgb_array",
        )
        self.num_envs = self._env.num_envs
        self.cfg = self._env.cfg
        self.device = self._env.device
        self.unwrapped = self._env

        self.cuda_stream = (
            connect_torch_to_mjlab(self._env.sim, torch_device)
            if torch_device.type == "cuda"
            else None
        )

        with self.compute_context():
            self._env.reset()

    def compute_context(self) -> AbstractContextManager[None]:
        return stream_context(self.cuda_stream)

    def robot_state(self) -> RobotState:
        data = self._env.scene["robot"].data
        return RobotState(
            root_pos_w=data.root_link_pos_w[0],
            root_quat_w=data.root_link_quat_w[0],
            root_ang_vel_b=data.root_link_ang_vel_b[0],
            projected_gravity_b=data.projected_gravity_b[0],
            joint_pos=data.joint_pos[0],
            joint_vel=data.joint_vel[0],
        )

    def get_observations(self) -> VecEnvObs:
        with self.compute_context():
            return self._env.get_observations()

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        with self.compute_context():
            return self._env.step(actions)

    def task_collision_detected(self) -> bool:
        """Whether a task obstacle was touched in the current physics state."""
        with self.compute_context():
            sim = self._env.sim
            data = sim.data
            active_contacts = min(int(data.nacon.item()), int(data.naconmax))
            if active_contacts == 0:
                return False

            # MJLab uses MuJoCo Warp's flat, batched contact pool. ``nacon``
            # gives its active length; each contact's two geom IDs are in
            # ``contact.geom``. The host model owns the corresponding names.
            model = sim._mj_model
            geom_pairs = data.contact.geom[:active_contacts].detach().cpu().tolist()
            for geom1, geom2 in geom_pairs:
                names = tuple(
                    mujoco.mj_id2name(  # ty: ignore[unresolved-attribute]
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,  # ty: ignore[unresolved-attribute]
                        int(geom_id),
                    )
                    for geom_id in (geom1, geom2)
                )
                if _is_robot_task_collision(*names):
                    return True
            return False

    def reset(self) -> tuple[VecEnvObs, dict[str, object]]:
        with self.compute_context():
            return self._env.reset()

    def reset_at(self, x: float, y: float) -> None:
        """Reset the robot at a world-frame XY position for a demo run."""
        with self.compute_context():
            self._env.reset()
            robot = self._env.scene["robot"].data
            root_state = robot.default_root_state.clone()
            root_state[:, 0] = x
            root_state[:, 1] = y
            robot.write_root_state(root_state)
            self._env.sim.forward()

    def render(self) -> np.ndarray:
        with self.compute_context():
            self._update_sokoban_completion_visuals()
            self._sync_offscreen_camera_heading()
            image = self._env.render()
        if image is None:
            raise RuntimeError("MJLab offscreen renderer returned no image")
        return image

    def render_rgbd(self) -> tuple[np.ndarray, ProjectionContext]:
        with self.compute_context():
            offline = self._env._offline_renderer
            if offline is None:
                raise RuntimeError("MJLab offscreen renderer is not initialized")
            self._update_sokoban_completion_visuals()
            self._sync_offscreen_camera_heading()
            debug_callback = (
                self._env.update_visualizers
                if hasattr(self._env, "update_visualizers")
                else None
            )
            offline.update(self._env.sim.data, debug_vis_callback=debug_callback)
            renderer = offline.renderer
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            try:
                depth = renderer.render().copy()
            finally:
                renderer.disable_depth_rendering()

            camera_pos = np.empty(3, dtype=np.float64)
            camera_forward = np.empty(3, dtype=np.float64)
            camera_up = np.empty(3, dtype=np.float64)
            mujoco.mjv_cameraInModel(  # ty: ignore[unresolved-attribute]
                camera_pos,
                camera_forward,
                camera_up,
                renderer.scene,
            )
            model = renderer.model
            extent = float(model.stat.extent)
            state = self.robot_state()
            projection = ProjectionContext(
                depth=depth,
                camera_pos_w=camera_pos,
                camera_forward_w=camera_forward,
                camera_up_w=camera_up,
                frustum_height=float(
                    mujoco.mjv_frustumHeight(  # ty: ignore[unresolved-attribute]
                        renderer.scene
                    )
                ),
                root_pos_w=state.root_pos_w.detach().cpu().numpy(),
                root_quat_w=state.root_quat_w.detach().cpu().numpy(),
                near=float(model.vis.map.znear) * extent,
                far=float(model.vis.map.zfar) * extent,
            )
        return rgb, projection

    def render_depth(self) -> ProjectionContext:
        with self.compute_context():
            offline = self._env._offline_renderer
            if offline is None:
                raise RuntimeError("MJLab offscreen renderer is not initialized")
            self._update_sokoban_completion_visuals()
            self._sync_offscreen_camera_heading()
            offline.update(self._env.sim.data)
            renderer = offline.renderer
            renderer.enable_depth_rendering()
            try:
                depth = renderer.render().copy()
            finally:
                renderer.disable_depth_rendering()
            camera_pos = np.empty(3, dtype=np.float64)
            camera_forward = np.empty(3, dtype=np.float64)
            camera_up = np.empty(3, dtype=np.float64)
            mujoco.mjv_cameraInModel(  # ty: ignore[unresolved-attribute]
                camera_pos, camera_forward, camera_up, renderer.scene
            )
            model = renderer.model
            extent = float(model.stat.extent)
            state = self.robot_state()
            return ProjectionContext(
                depth=depth,
                camera_pos_w=camera_pos,
                camera_forward_w=camera_forward,
                camera_up_w=camera_up,
                frustum_height=float(
                    mujoco.mjv_frustumHeight(  # ty: ignore[unresolved-attribute]
                        renderer.scene
                    )
                ),
                root_pos_w=state.root_pos_w.detach().cpu().numpy(),
                root_quat_w=state.root_quat_w.detach().cpu().numpy(),
                near=float(model.vis.map.znear) * extent,
                far=float(model.vis.map.zfar) * extent,
            )

    def render_demo_rgb(self) -> np.ndarray:
        """Render the same offscreen camera view that is sent to the VLM."""
        with self.compute_context():
            offline = self._env._offline_renderer
            if offline is None:
                raise RuntimeError("MJLab offscreen renderer is not initialized")
            self._update_sokoban_completion_visuals()
            self._sync_offscreen_camera_heading()
            offline.update(self._env.sim.data)
            return offline.render().copy()

    def _update_sokoban_completion_visuals(self) -> None:
        """Color a box green only after it is safely inside its own goal."""

        if self._task is None or self._task.name != "sokoban":
            return
        offline = self._env._offline_renderer
        if offline is None:
            return

        from tasks.sokoban.scene import (
            BOX_STARTS,
            COMPLETED_BOX_RGBA,
            _BOX_RGBA,
            completed_box_indices,
        )

        sim = self._env.sim
        model = sim.mj_model
        qpos = sim.data.qpos[0].detach().cpu().numpy()
        centers = tuple(
            (
                start_x + float(qpos[model.joint(f"sokoban_box_{index}_x").qposadr]),
                start_y + float(qpos[model.joint(f"sokoban_box_{index}_y").qposadr]),
            )
            for index, (start_x, start_y) in enumerate(BOX_STARTS, start=1)
        )
        for index, complete in enumerate(completed_box_indices(centers), start=1):
            geom_id = model.geom(f"sokoban_box_{index}_pushable").id
            color = COMPLETED_BOX_RGBA if complete else _BOX_RGBA
            model.geom_rgba[geom_id] = color
            offline.renderer.model.geom_rgba[geom_id] = color

    def _sync_offscreen_camera_heading(self) -> None:
        if not getattr(self, "_camera_yaw", True):
            return
        offline = self._env._offline_renderer
        if offline is None:
            return
        root_quat_w = (
            self._env.scene["robot"].data.root_link_quat_w[0].detach().cpu().numpy()
        )
        offline._cam.azimuth = _heading_camera_azimuth(
            self._camera_base_azimuth, root_quat_w
        )

    def close(self) -> None:
        self._env.close()


# CUDA


def connect_torch_to_mjlab(
    simulation: Simulation,
    device: torch.device,
) -> torch.cuda.Stream:
    import warp as wp

    torch.cuda.synchronize(device)
    wp.synchronize_device(simulation.wp_device)
    return wp.stream_to_torch(simulation.wp_device)


def stream_context(stream: torch.cuda.Stream | None) -> AbstractContextManager[None]:
    return torch.cuda.stream(stream) if stream is not None else nullcontext()


def _is_robot_task_collision(name1: str | None, name2: str | None) -> bool:
    """Whether a contact is between the robot and a notification obstacle."""

    return (_is_robot_geom(name1) and _is_task_collision_geom(name2)) or (
        _is_robot_geom(name2) and _is_task_collision_geom(name1)
    )


def _is_robot_geom(name: str | None) -> bool:
    return name is not None and name.startswith("robot/")


def _is_task_collision_geom(name: str | None) -> bool:
    return (
        name is not None
        and not name.startswith("robot/")
        and name.endswith("_collision")
    )


def _heading_camera_azimuth(base_azimuth: float, root_quat_w: np.ndarray) -> float:
    return base_azimuth + math.degrees(yaw_from_quat_wxyz(root_quat_w))
