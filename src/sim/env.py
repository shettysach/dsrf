from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensor
from tasks import TaskSpec

from sim.camera import OnDemandCameraCapture, ProjectionContext
from sim.config import OBSERVATION_CAMERA, make_sim_env_cfg
from tracker.state import RobotState

if TYPE_CHECKING:
    from mjlab.envs.types import VecEnvStepReturn
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
            render_mode=None,
        )
        self.unwrapped = self._env

        self._robot = self._env.scene["robot"]
        camera = self._env.scene[OBSERVATION_CAMERA]
        assert isinstance(camera, CameraSensor)
        self._camera_capture = OnDemandCameraCapture(
            self._env.sim,
            camera,
        )
        self.cuda_stream = (
            connect_torch_to_mjlab(self._env.sim, torch_device)
            if torch_device.type == "cuda"
            else None
        )

        with self.compute_context():
            self._env.reset()

    @property
    def cfg(self):
        return self._env.cfg

    @property
    def device(self) -> torch.device | str:
        return self._env.device

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    def compute_context(self) -> AbstractContextManager[None]:
        return stream_context(self.cuda_stream)

    def robot_state(self) -> RobotState:
        data = self._robot.data
        return RobotState(
            root_pos_w=data.root_link_pos_w[0],
            root_quat_w=data.root_link_quat_w[0],
            root_ang_vel_b=data.root_link_ang_vel_b[0],
            projected_gravity_b=data.projected_gravity_b[0],
            joint_pos=data.joint_pos[0],
            joint_vel=data.joint_vel[0],
        )

    @property
    def step_dt(self) -> float:
        return float(self._env.step_dt)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        with self.compute_context():
            return self._env.step(action)

    def capture_rgbd(self) -> tuple[torch.Tensor, ProjectionContext]:
        with self.compute_context():
            return self._camera_capture.capture(self.robot_state())

    def close(self) -> None:
        self._camera_capture.close()
        self._env.close()


# CUDA


def connect_torch_to_mjlab(
    simulation: Simulation,
    device: torch.device,
) -> torch.cuda.Stream:
    import warp as wp

    return wp.stream_to_torch(simulation.wp_device)


def stream_context(stream: torch.cuda.Stream | None) -> AbstractContextManager[None]:
    return torch.cuda.stream(stream) if stream is not None else nullcontext()
