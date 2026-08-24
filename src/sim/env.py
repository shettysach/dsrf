from __future__ import annotations

import re
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensor
from tasks import TaskSpec

from controller import ContactDynamics, ControlOutput, DynamicsSnapshot, RobotState
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
        self._dynamics_data = mujoco.MjData(self._env.sim.mj_model)
        self._joint_stiffness, self._joint_damping, self._effort_limits = (
            self._pd_parameters()
        )

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

    def dynamics_snapshot(self, state: RobotState | None = None) -> DynamicsSnapshot:
        """Extract shadow MuJoCo dynamics for exactly this controller state.

        MJLab owns the live physics state.  A local MuJoCo data object is used
        only to evaluate its model functions at the RobotState sampled in the
        current compute context; it is never stepped by this wrapper.
        """
        state = self.robot_state() if state is None else state
        model = self._env.sim.mj_model
        data = self._dynamics_data
        indexing = self._robot.indexing
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        free_q = indexing.free_joint_q_adr.cpu().numpy()
        joint_q = indexing.joint_q_adr.cpu().numpy()
        free_v = indexing.free_joint_v_adr.cpu().numpy()
        joint_v = indexing.joint_v_adr.cpu().numpy()
        data.qpos[free_q[:3]] = state.root_pos_w.detach().cpu().numpy()
        data.qpos[free_q[3:]] = state.root_quat_w.detach().cpu().numpy()
        data.qpos[joint_q] = state.joint_pos.detach().cpu().numpy()
        data.qvel[free_v[:3]] = state.root_lin_vel_w.detach().cpu().numpy()
        data.qvel[free_v[3:]] = state.root_ang_vel_w.detach().cpu().numpy()
        data.qvel[joint_v] = state.joint_vel.detach().cpu().numpy()
        mujoco.mj_forward(model, data)
        mass = np.empty((model.nv, model.nv), dtype=np.float64)
        mujoco.mj_fullM(model, data, mass)
        contacts = tuple(self._support_contacts(data))
        device = state.joint_pos.device
        return DynamicsSnapshot(
            qpos=torch.as_tensor(data.qpos.copy(), dtype=torch.float64, device=device),
            qvel=torch.as_tensor(data.qvel.copy(), dtype=torch.float64, device=device),
            mass_matrix=torch.as_tensor(mass, dtype=torch.float64, device=device),
            bias_force=torch.as_tensor(
                data.qfrc_bias.copy(), dtype=torch.float64, device=device
            ),
            contacts=contacts,
            actuated_dof_indices=torch.as_tensor(
                joint_v, dtype=torch.long, device=device
            ),
            joint_stiffness=self._joint_stiffness.to(device=device, dtype=torch.float64),
            joint_damping=self._joint_damping.to(device=device, dtype=torch.float64),
            effort_limits=self._effort_limits.to(device=device, dtype=torch.float64),
        )

    def _support_contacts(self, data: object) -> list[ContactDynamics]:
        model = self._env.sim.mj_model
        contacts: list[ContactDynamics] = []
        selected_bodies: set[str] = set()
        support_bodies = {"left_ankle_roll_link", "right_ankle_roll_link"}
        for index in range(data.ncon):  # ty: ignore[unresolved-attribute]
            contact = data.contact[index]  # ty: ignore[unresolved-attribute]
            body_ids = (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2])
            names = tuple(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
                for body_id in body_ids
            )
            robot_name = next(
                (name for name in names if name and name.removeprefix("robot/") in support_bodies),
                None,
            )
            if robot_name is None or not any(name == "terrain" for name in names):
                continue
            local_name = robot_name.removeprefix("robot/")
            # MuJoCo usually reports many geom pairs under one foot.  One
            # point per support body keeps the equality system independent in
            # this deliberately minimal v1 controller.
            if local_name in selected_bodies:
                continue
            selected_bodies.add(local_name)
            point = np.asarray(contact.pos, dtype=np.float64)
            jacobian = np.empty((3, model.nv), dtype=np.float64)
            jacobian_dot = np.empty((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(model, data, jacobian, None, point, int(body_ids[names.index(robot_name)]))
            mujoco.mj_jacDot(model, data, jacobian_dot, None, point, int(body_ids[names.index(robot_name)]))
            device = self._joint_stiffness.device
            contacts.append(
                ContactDynamics(
                    body=local_name,
                    position_w=torch.as_tensor(point, dtype=torch.float64, device=device),
                    frame_w=torch.as_tensor(
                        np.asarray(contact.frame, dtype=np.float64).reshape(3, 3),
                        dtype=torch.float64,
                        device=device,
                    ),
                    jacobian=torch.as_tensor(jacobian, dtype=torch.float64, device=device),
                    jacobian_dot_velocity=torch.as_tensor(
                        jacobian_dot @ data.qvel, dtype=torch.float64, device=device
                    ),
                )
            )
        return contacts

    def _pd_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the exact configured PD parameters in canonical joint order."""
        cfgs = self.cfg.scene.entities["robot"].articulation.actuators
        stiffness, damping, effort = [], [], []
        for joint in G1_JOINT_NAMES_MJLAB:
            actuator = next(
                (item for item in cfgs if any(re.fullmatch(pattern, joint) for pattern in item.target_names_expr)),
                None,
            )
            if actuator is None or actuator.effort_limit is None:
                raise RuntimeError(f"No PD actuator configuration for {joint}")
            stiffness.append(float(actuator.stiffness))
            damping.append(float(actuator.damping))
            effort.append(float(actuator.effort_limit))
        return tuple(torch.tensor(values, dtype=torch.float64) for values in (stiffness, damping, effort))  # ty: ignore[invalid-return-type]

    @property
    def step_dt(self) -> float:
        return float(self._env.step_dt)


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
