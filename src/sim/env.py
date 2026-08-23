from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from tasks import TaskSpec

from controller.types import RobotState
from shared.messages import ProjectionContext
from sim.config import make_sim_env_cfg

if TYPE_CHECKING:
    from mjlab.envs.types import VecEnvObs, VecEnvStepReturn
    from mjlab.sim import Simulation


class MjlabEnv:
    def __init__(
        self,
        *,
        device: str = "cpu",
        image_width: int = 640,
        image_height: int = 480,
        task: TaskSpec | None = None,
    ) -> None:
        torch_device = torch.device(device)
        self._env = ManagerBasedRlEnv(
            cfg=make_sim_env_cfg(
                image_width=image_width,
                image_height=image_height,
                task=task,
            ),
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

    def reset(self) -> tuple[VecEnvObs, dict[str, object]]:
        with self.compute_context():
            return self._env.reset()

    def render(self) -> np.ndarray:
        with self.compute_context():
            renderer = self._update_offscreen_renderer()
            return renderer.render().copy()

    def render_depth(self) -> ProjectionContext:
        with self.compute_context():
            renderer = self._update_offscreen_renderer()
            renderer.enable_depth_rendering()
            try:
                depth = renderer.render().copy()
            finally:
                renderer.disable_depth_rendering()
            return self._projection_context(renderer, depth)

    def render_rgbd(self) -> tuple[np.ndarray, ProjectionContext]:
        with self.compute_context():
            renderer = self._update_offscreen_renderer()
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            try:
                depth = renderer.render().copy()
            finally:
                renderer.disable_depth_rendering()

            projection = self._projection_context(renderer, depth)
        return rgb, projection

    def _update_offscreen_renderer(self) -> Any:
        offline = self._env._offline_renderer
        if offline is None:
            raise RuntimeError("MJLab offscreen renderer is not initialized")
        debug_callback = (
            self._env.update_visualizers
            if hasattr(self._env, "update_visualizers")
            else None
        )
        offline.update(self._env.sim.data, debug_vis_callback=debug_callback)
        return offline.renderer

    def _projection_context(
        self, renderer: Any, depth: np.ndarray
    ) -> ProjectionContext:
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
