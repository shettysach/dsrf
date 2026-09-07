from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
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
        self._device = torch_device
        self._task = task
        self._env = ManagerBasedRlEnv(
            cfg=make_sim_env_cfg(
                image_width=image_width,
                image_height=image_height,
                task=task,
            ),
            device=str(torch_device),
            render_mode=None,
        )
        self._robot = self._env.scene["robot"]
        self._hand_geom_ids = _hand_geom_ids(self._robot)
        self._object_geom_ids = _object_geom_ids(
            self._env.scene,
            tuple(
                name
                for name in self._env.scene.entities
                if name not in {"robot", "terrain"}
            ),
        )
        self._virtual_force_object_geom_ids = _object_geom_ids(
            self._env.scene,
            task.virtual_force_objects if task is not None else (),
        )
        self._virtual_force_body_ids = {
            name: _single_geom_body_id(
                self._env.sim.mj_model,
                geom_ids,
                object_name=name,
            )
            for name, geom_ids in self._virtual_force_object_geom_ids.items()
        }
        camera = self._env.scene[OBSERVATION_CAMERA]
        assert isinstance(camera, CameraSensor)
        self._camera_capture = OnDemandCameraCapture(
            self._env.sim,
            camera,
            follow_robot_translation=(
                task is not None
                and task.observation_camera.follow_robot_translation
            ),
        )
        self._sokoban_visualizer = None
        self._task_completed = False
        if task is not None and task.name == "sokoban":
            from tasks.sokoban.scene import SokobanCompletionVisualizer

            self._sokoban_visualizer = SokobanCompletionVisualizer(
                self._env.sim.mj_model
            )
        self.cuda_stream = (
            connect_torch_to_mjlab(self._env.sim, torch_device)
            if torch_device.type == "cuda"
            else None
        )

        with self.compute_context():
            self._env.reset()
            self.update_task_visuals()

    @property
    def mjlab_env(self) -> ManagerBasedRlEnv:
        """The native environment for MJLab-owned integrations such as viewers."""
        return self._env

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def task(self) -> TaskSpec | None:
        return self._task

    @property
    def task_completed(self) -> bool:
        """Whether the task's optional physical completion check has passed."""
        return self._task_completed

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

    def hand_object_contacts(
        self,
        object_names: tuple[str, ...],
    ) -> set[tuple[str, str]]:
        """Return only left/right-hand contacts with named task entities."""

        requested = set(object_names)
        unknown = requested - self._object_geom_ids.keys()
        if unknown:
            raise ValueError(f"Unknown contact objects: {sorted(unknown)}")
        data = self._env.sim.data
        return _hand_object_contacts_from_buffers(
            geom_pairs=data.contact.geom,
            world_ids=data.contact.worldid,
            contact_count=data.nacon[0],
            hand_geom_ids=self._hand_geom_ids,
            object_geom_ids={name: self._object_geom_ids[name] for name in requested},
        )

    def push_state(self, body: str):
        """Snapshot physical state for the scripted single-box controller."""
        from sim.push import PushState

        entity = self._env.scene[body]
        if not isinstance(entity, Entity):
            raise ValueError("Scripted push requires an MJLab entity")
        geom_ids = sorted(self._object_geom_ids[body])
        model = self._env.sim.mj_model
        if len(geom_ids) != 1 or int(model.geom_type[geom_ids[0]]) != 6:
            raise ValueError("Scripted push currently requires one box geometry")
        state = self.robot_state()

        def cpu(tensor):
            return tensor.detach().cpu().numpy().copy()

        half_size = np.asarray(model.geom_size[geom_ids[0]])
        corners = (
            np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            * half_size
        )
        sim_data = self._env.sim.data
        geom_rotation = cpu(sim_data.geom_xmat[0, geom_ids[0]]).reshape(3, 3)
        geom_position = cpu(sim_data.geom_xpos[0, geom_ids[0]])
        geom_body_id = int(model.geom_bodyid[geom_ids[0]])
        return PushState(
            qpos=cpu(torch.cat((state.root_pos_w, state.root_quat_w, state.joint_pos))),
            body_position=geom_position,
            body_rotation=geom_rotation,
            body_velocity=cpu(sim_data.cvel[0, geom_body_id, 3:6]),
            hands={
                name: cpu(sim_data.geom_xpos[0, gid])
                for name, gid in self._hand_geom_ids.items()
            },
            contacts=frozenset(hand for hand, _ in self.hand_object_contacts((body,))),
            footprint=(corners @ geom_rotation.T + geom_position)[:, :2],
        )

    def step(
        self,
        action: torch.Tensor,
        *,
        external_forces: Mapping[str, torch.Tensor] | None = None,
    ) -> VecEnvStepReturn:
        with self.compute_context():
            self._write_external_forces(external_forces or {})
            result = self._env.step(action)
            self.update_task_visuals()
            return result

    def update_task_visuals(self) -> None:
        """Refresh dynamic task-only rendering state after simulation changes."""
        if self._sokoban_visualizer is not None:
            self._task_completed = self._sokoban_visualizer.update(self._env.sim.data)

    def capture_rgbd(self) -> tuple[torch.Tensor, ProjectionContext]:
        with self.compute_context():
            return self._camera_capture.capture(self.robot_state())

    def close(self) -> None:
        self._camera_capture.close()
        self._env.close()

    def _write_external_forces(self, forces: Mapping[str, torch.Tensor]) -> None:
        unknown = set(forces) - self._virtual_force_body_ids.keys()
        if unknown:
            raise ValueError(f"Unknown virtual-force objects: {sorted(unknown)}")
        for name, body_id in self._virtual_force_body_ids.items():
            force = forces.get(name)
            if force is None:
                force = torch.zeros(3, dtype=torch.float32, device=self._device)
            if force.shape != (3,):
                raise ValueError(
                    f"Virtual force for {name!r} must have shape (3,), got "
                    f"{tuple(force.shape)}"
                )
            force = force.to(device=self._device, dtype=torch.float32)
            data = self._env.sim.data
            data.xfrc_applied[:, body_id, 0:3] = force.reshape(1, 3)
            data.xfrc_applied[:, body_id, 3:6] = 0.0


def _hand_geom_ids(robot: Entity) -> dict[str, int]:
    geom_ids = robot.indexing.geom_ids.detach().cpu().tolist()
    geom_ids_by_name = dict(zip(robot.geom_names, geom_ids, strict=True))
    return {
        hand: geom_ids_by_name[geom_name]
        for hand, geom_name in {
            "left_hand": "left_hand_collision",
            "right_hand": "right_hand_collision",
        }.items()
    }


def _object_geom_ids(
    scene,
    object_names: tuple[str, ...],
) -> dict[str, frozenset[int]]:
    object_geoms: dict[str, frozenset[int]] = {}
    for name in object_names:
        entity = scene[name]
        if not isinstance(entity, Entity):
            raise ValueError(f"Virtual-force object {name!r} must be an MJLab Entity")
        object_geoms[name] = frozenset(entity.indexing.geom_ids.detach().cpu().tolist())
    return object_geoms


def _single_geom_body_id(model, geom_ids: frozenset[int], *, object_name: str) -> int:
    if len(geom_ids) != 1:
        raise ValueError(
            f"Virtual-force object {object_name!r} must have exactly one collision geom"
        )
    return int(model.geom_bodyid[next(iter(geom_ids))])


def _hand_object_contacts_from_buffers(
    *,
    geom_pairs: torch.Tensor,
    world_ids: torch.Tensor,
    contact_count: torch.Tensor,
    hand_geom_ids: dict[str, int],
    object_geom_ids: dict[str, frozenset[int]],
) -> set[tuple[str, str]]:
    """Find eligible contacts without copying MJWarp contact buffers to the CPU."""

    slot_indices = torch.arange(geom_pairs.shape[0], device=geom_pairs.device)
    active_contacts = (slot_indices < contact_count) & (world_ids == 0)
    geom_a, geom_b = geom_pairs.unbind(dim=-1)
    contacts: set[tuple[str, str]] = set()
    for hand, hand_geom_id in hand_geom_ids.items():
        hand_is_a = geom_a == hand_geom_id
        hand_is_b = geom_b == hand_geom_id
        for object_name, object_ids in object_geom_ids.items():
            object_id_tensor = torch.tensor(
                tuple(object_ids),
                device=geom_pairs.device,
                dtype=geom_pairs.dtype,
            )
            matches = active_contacts & (
                (hand_is_a & torch.isin(geom_b, object_id_tensor))
                | (hand_is_b & torch.isin(geom_a, object_id_tensor))
            )
            # The public API returns symbolic names, so only this final scalar
            # crosses into Python; all contact-buffer filtering remains on-device.
            if matches.any().item():
                contacts.add((hand, object_name))
    return contacts


# CUDA


def connect_torch_to_mjlab(
    simulation: Simulation,
    device: torch.device,
) -> torch.cuda.Stream:
    import warp as wp

    return wp.stream_to_torch(simulation.wp_device)


def stream_context(stream: torch.cuda.Stream | None) -> AbstractContextManager[None]:
    return torch.cuda.stream(stream) if stream is not None else nullcontext()
