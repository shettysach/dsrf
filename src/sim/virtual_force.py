"""Object-only virtual assistance from contact or coarse palm proximity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul

from tracker.state import RobotState

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mujoco import MjModel  # ty: ignore[unresolved-import]


VIRTUAL_FORCE_BODIES = frozenset(("left_hand", "right_hand"))
HAND_GEOM_NAMES = {
    "left_hand": "left_hand_collision",
    "right_hand": "right_hand_collision",
}

MIN_HAND_SPEED = 0.05
LOOKAHEAD_FRAMES = 3
RAMP_FRAMES = 4

ContactPair = tuple[str, str]


@dataclass(frozen=True)
class VirtualForceResult:
    forces: dict[str, torch.Tensor]
    started_contacts: tuple[ContactPair, ...]
    ended_contacts: tuple[ContactPair, ...]


class ProximityPushForce:
    """Assist the scripted +X box push while a palm stays near its rear face."""

    def __init__(
        self,
        *,
        half_size: tuple[float, float, float],
        goal_x: float,
        magnitude: float,
        enable_distance: float,
        disable_distance: float,
        device: torch.device | str,
    ) -> None:
        if not 0.0 < enable_distance < disable_distance:
            raise ValueError("Push-force distances must satisfy 0 < enable < disable")
        self.half_size = np.asarray(half_size)
        self.goal_x = goal_x
        self.magnitude = magnitude
        self.enable_distance = enable_distance
        self.disable_distance = disable_distance
        self.device = torch.device(device)
        self.active = False
        self.last_distance = float("inf")
        self.last_force = torch.zeros(3, dtype=torch.float32, device=self.device)

    def compute(
        self,
        phase: str,
        box_position: np.ndarray,
        hands: dict[str, np.ndarray],
    ) -> VirtualForceResult:
        box = np.asarray(box_position)
        half_x, half_y, half_z = self.half_size
        front_x = box[0] - half_x
        self.last_distance = min(
            (
                float(
                    np.linalg.norm(
                        (
                            hand[0] - front_x,
                            max(abs(hand[1] - box[1]) - half_y, 0.0),
                            max(abs(hand[2] - box[2]) - half_z, 0.0),
                        )
                    )
                )
                for hand in hands.values()
            ),
            default=float("inf"),
        )
        if phase != "push" or box[0] >= self.goal_x - 0.10:
            self.active = False
        elif self.active:
            self.active = self.last_distance <= self.disable_distance
        else:
            self.active = self.last_distance < self.enable_distance

        self.last_force = torch.tensor(
            (self.magnitude if self.active else 0.0, 0.0, 0.0),
            dtype=torch.float32,
            device=self.device,
        )
        return VirtualForceResult({"box": self.last_force}, (), ())


@dataclass(frozen=True)
class _HandMotion:
    directions: torch.Tensor
    speeds: torch.Tensor


class VirtualForce:
    """Produce object forces only while an eligible hand-object pair touches."""

    def __init__(
        self,
        object_names: Iterable[str],
        *,
        dt: float,
        device: torch.device | str,
        magnitude: float,
        maximum: float,
        min_hand_speed: float = MIN_HAND_SPEED,
        lookahead_frames: int = LOOKAHEAD_FRAMES,
        ramp_frames: int = RAMP_FRAMES,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("Virtual-force timestep must be positive")
        if magnitude < 0.0 or maximum < 0.0:
            raise ValueError("Virtual-force magnitudes must be non-negative")
        if min_hand_speed < 0.0:
            raise ValueError("Minimum hand speed must be non-negative")
        if lookahead_frames < 1:
            raise ValueError("Virtual-force lookahead must be positive")
        if ramp_frames < 1:
            raise ValueError("Virtual-force ramp frames must be positive")

        self.object_names = tuple(object_names)
        self.device = torch.device(device)
        self.dt = dt
        self.magnitude = magnitude
        self.maximum = maximum
        self.min_hand_speed = min_hand_speed
        self.lookahead_frames = lookahead_frames
        self.ramp_frames = ramp_frames
        self._hand_motion: dict[str, _HandMotion] = {}
        self._contact_ages: dict[ContactPair, int] = {}
        self._fk_model: MjModel | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.object_names)

    def load_motion(self, qpos: torch.Tensor, state: RobotState) -> None:
        """Precompute world-aligned hand direction and speed for a reference."""

        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(
                f"Expected reference qpos [T, 36], got {tuple(qpos.shape)}"
            )
        if len(qpos) < 2:
            raise ValueError("Virtual force requires at least two reference frames")

        qpos = qpos.to(device=self.device, dtype=torch.float32).contiguous()
        hand_positions = self._forward_kinematics(qpos)
        orientation_delta = quat_mul(state.root_quat_w, quat_conjugate(qpos[0, 3:7]))
        root_relative_positions = {
            hand: positions - qpos[0, :3] for hand, positions in hand_positions.items()
        }
        self._hand_motion = {
            hand: _derive_hand_motion(
                state.root_pos_w
                + quat_apply(
                    orientation_delta.expand(len(root_relative), -1), root_relative
                ),
                dt=self.dt,
                lookahead_frames=self.lookahead_frames,
            )
            for hand, root_relative in root_relative_positions.items()
        }
        self._contact_ages.clear()

    def compute(
        self,
        frame: int,
        contacts: set[ContactPair],
    ) -> VirtualForceResult:
        """Return per-object assistance for the current reference frame."""

        if not self._hand_motion:
            raise RuntimeError("Virtual-force motion has not been loaded")
        if frame < 0:
            raise ValueError("Reference frame must be non-negative")

        active_contacts = {
            (hand, object_name)
            for hand, object_name in contacts
            if hand in VIRTUAL_FORCE_BODIES and object_name in self.object_names
        }
        previous_contacts = set(self._contact_ages)
        started = tuple(sorted(active_contacts - previous_contacts))
        ended = tuple(sorted(previous_contacts - active_contacts))

        forces = {
            object_name: torch.zeros(3, dtype=torch.float32, device=self.device)
            for object_name in self.object_names
        }
        next_ages: dict[ContactPair, int] = {}
        for hand, object_name in sorted(active_contacts):
            age = self._contact_ages.get((hand, object_name), 0)
            next_ages[(hand, object_name)] = age + 1
            motion = self._hand_motion[hand]
            motion_frame = min(frame, len(motion.speeds) - 1)
            if motion.speeds[motion_frame] < self.min_hand_speed:
                continue
            ramp = min(age / self.ramp_frames, 1.0)
            forces[object_name] += (
                ramp * self.magnitude * motion.directions[motion_frame]
            )

        self._contact_ages = next_ages
        for object_name, force in forces.items():
            force_norm = torch.linalg.vector_norm(force)
            if force_norm > self.maximum:
                forces[object_name] = force * (self.maximum / force_norm)

        return VirtualForceResult(forces, started, ended)

    def _forward_kinematics(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        model = self._get_fk_model()
        data = mujoco.MjData(model)  # ty: ignore[unresolved-attribute]
        geom_ids = {hand: model.geom(name).id for hand, name in HAND_GEOM_NAMES.items()}
        positions = {
            hand: np.empty((len(qpos), 3), dtype=np.float32) for hand in HAND_GEOM_NAMES
        }
        qpos_cpu = qpos.cpu().numpy()
        for frame, frame_qpos in enumerate(qpos_cpu):
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)  # ty: ignore[unresolved-attribute]
            for hand, geom_id in geom_ids.items():
                positions[hand][frame] = data.geom_xpos[geom_id]
        return {
            hand: torch.as_tensor(values, dtype=torch.float32, device=self.device)
            for hand, values in positions.items()
        }

    def _get_fk_model(self) -> MjModel:
        if self._fk_model is None:
            self._fk_model = get_g1_robot_cfg().build().spec.compile()
        return self._fk_model


def _derive_hand_motion(
    positions: torch.Tensor,
    *,
    dt: float,
    lookahead_frames: int,
) -> _HandMotion:
    frame_count = len(positions)
    indices = torch.arange(frame_count, device=positions.device)
    next_indices = torch.clamp(indices + lookahead_frames, max=frame_count - 1)
    steps = next_indices - indices
    delta = positions.index_select(0, next_indices) - positions
    distance = torch.linalg.vector_norm(delta, dim=-1)
    speeds = torch.zeros_like(distance)
    nonzero_steps = steps > 0
    speeds[nonzero_steps] = distance[nonzero_steps] / (steps[nonzero_steps] * dt)
    directions = torch.zeros_like(delta)
    nonzero_distance = distance > 0.0
    directions[nonzero_distance] = delta[nonzero_distance] / distance[
        nonzero_distance
    ].unsqueeze(-1)
    return _HandMotion(directions=directions, speeds=speeds)
