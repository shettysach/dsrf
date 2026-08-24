from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensor
from tasks import TaskSpec

from controller import ControlOutput, RobotState
from controller.g1_command import G1CommandTransform
from shared.g1 import G1_JOINT_COUNT, G1_JOINT_NAMES_MJLAB
from sim.camera import OnDemandCameraCapture, ProjectionContext
from sim.config import OBSERVATION_CAMERA, ControlMode, make_sim_env_cfg

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
        control_mode: ControlMode = "position",
    ) -> None:
        torch_device = torch.device(device)
        self._env = ManagerBasedRlEnv(
            cfg=make_sim_env_cfg(
                image_width=image_width,
                image_height=image_height,
                task=task,
                control_mode=control_mode,
            ),
            device=str(torch_device),
            render_mode=None,
        )
        self.num_envs = self._env.num_envs
        self.cfg = self._env.cfg
        self.device = self._env.device
        self.unwrapped = self._env
        self.control_mode = control_mode

        action_term = self._env.action_manager.get_term("joint_position")
        if tuple(action_term.target_names) != G1_JOINT_NAMES_MJLAB:  # ty: ignore[unresolved-attribute]
            raise RuntimeError(
                "MJLab joint action order does not match canonical G1 order"
            )
        if action_term.action_dim != G1_JOINT_COUNT:  # ty: ignore[unresolved-attribute]
            raise RuntimeError("MJLab joint position action must have 29 targets")
        self.command_transform = G1CommandTransform(
            action_term.offset[0],  # ty: ignore[unresolved-attribute]
            action_term.scale[0],  # ty: ignore[unresolved-attribute]
        )
        if self.control_mode == "pd":
            velocity_term = self._env.action_manager.get_term("joint_velocity")
            if tuple(velocity_term.target_names) != G1_JOINT_NAMES_MJLAB:  # ty: ignore[unresolved-attribute]
                raise RuntimeError(
                    "MJLab joint velocity action order does not match canonical G1 order"
                )
            if velocity_term.action_dim != G1_JOINT_COUNT:  # ty: ignore[unresolved-attribute]
                raise RuntimeError("MJLab joint velocity action must have 29 targets")
        self._robot = self._env.scene["robot"]
        camera = self._env.scene[OBSERVATION_CAMERA]
        assert isinstance(camera, CameraSensor)
        self._camera_capture = OnDemandCameraCapture(
            self._env.sim,
            camera,
        )
        self._body_ids = {
            name: index for index, name in enumerate(self._robot.body_names)
        }
        self._wrench_forces = torch.zeros(
            (self.num_envs, self._robot.num_bodies, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._wrench_torques = torch.zeros_like(self._wrench_forces)

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
        data = self._robot.data
        return RobotState(
            root_pos_w=data.root_link_pos_w[0],
            root_quat_w=data.root_link_quat_w[0],
            root_lin_vel_w=data.root_link_lin_vel_w[0],
            root_ang_vel_w=data.root_link_ang_vel_w[0],
            root_ang_vel_b=data.root_link_ang_vel_b[0],
            projected_gravity_b=data.projected_gravity_b[0],
            joint_pos=data.joint_pos[0],
            joint_vel=data.joint_vel[0],
        )

    @property
    def step_dt(self) -> float:
        return float(self._env.step_dt)

    @property
    def robot_mass(self) -> float:
        """Total mass of the configured G1 bodies from the MJLab model."""
        model = self._env.sim.mj_model
        return float(
            sum(model.body_mass[model.body(name).id] for name in self._robot.body_names)
        )

    @property
    def gravity_magnitude(self) -> float:
        return float(torch.linalg.vector_norm(torch.as_tensor(self._env.sim.mj_model.opt.gravity)))

    def step(self, output: ControlOutput) -> VecEnvStepReturn:
        with self.compute_context():
            self._apply_wrenches(output)
            return self._env.step(self._control_to_action(output))

    def _control_to_action(self, output: ControlOutput) -> torch.Tensor:
        """Translate a physical controller command to this environment's action API."""
        position_action = self.command_transform.encode(output.joint_target)
        if self.control_mode == "position":
            if output.joint_velocity_target is not None:
                raise ValueError(
                    "Position control mode does not accept velocity targets"
                )
            return position_action.unsqueeze(0)

        velocity_action = (
            output.joint_velocity_target
            if output.joint_velocity_target is not None
            else torch.zeros_like(output.joint_target)
        )
        return torch.cat((position_action, velocity_action)).unsqueeze(0)

    def _apply_wrenches(self, output: ControlOutput) -> None:
        self._wrench_forces.zero_()
        self._wrench_torques.zero_()
        requested: set[str] = set()
        for wrench in output.external_wrenches:
            if wrench.body in requested:
                raise ValueError(f"Duplicate external wrench for body {wrench.body!r}")
            requested.add(wrench.body)
            try:
                body_id = self._body_ids[wrench.body]
            except KeyError:
                raise ValueError(
                    f"Unknown external wrench body {wrench.body!r}"
                ) from None
            self._wrench_forces[:, body_id].copy_(wrench.force_w)
            self._wrench_torques[:, body_id].copy_(wrench.torque_w)
        self._robot.write_external_wrench_to_sim(
            self._wrench_forces,
            self._wrench_torques,
        )

    def reset(self) -> tuple[VecEnvObs, dict[str, object]]:
        with self.compute_context():
            return self._env.reset()

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
