"""Motion-reference alignment and traversal for SONIC tracking."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_conjugate,
    quat_mul,
)

from shared.g1 import standing_qpos
from shared.messages import REFERENCE_HZ, MotionChunk


class MotionReference:
    """A motion chunk aligned to the robot pose at command start."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        initial = torch.as_tensor(
            standing_qpos()[None], dtype=torch.float32, device=self.device
        )
        self._root_pos_w = initial[:, :3]
        self._root_quat_w = initial[:, 3:7]
        self._joint_pos = initial[:, 7:]
        self._joint_vel = torch.zeros((1, 29), device=self.device)
        self._frame = 0
        self._active = False

    def load(
        self,
        chunk: MotionChunk,
        robot_pos_w: torch.Tensor,
        robot_quat_w: torch.Tensor,
    ) -> None:
        qpos = torch.as_tensor(
            chunk.qpos, dtype=torch.float32, device=self.device
        ).contiguous()
        orientation_delta = quat_mul(robot_quat_w, quat_conjugate(qpos[0, 3:7]))
        root_delta = qpos[:, :3] - qpos[0, :3]
        self._root_pos_w = robot_pos_w + quat_apply(
            orientation_delta.expand(len(qpos), -1), root_delta
        )
        self._root_quat_w = quat_mul(
            orientation_delta.expand(len(qpos), -1), qpos[:, 3:7]
        )
        self._joint_pos = qpos[:, 7:]
        self._joint_vel = _finite_difference(self._joint_pos)
        self._frame = 0
        self._active = True

    @property
    def frame_count(self) -> int:
        return len(self._joint_pos)

    @property
    def frame_index(self) -> int:
        return self._frame

    def visualization_pose(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self._active:
            return None
        return (
            self._root_pos_w[self._frame],
            self._root_quat_w[self._frame],
            self._joint_pos[self._frame],
        )

    def window(
        self, *, count: int, step: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets = torch.arange(count, dtype=torch.long, device=self.device)
        indices = torch.clamp(offsets * step + self._frame, max=self.frame_count - 1)
        return (
            self._joint_pos.index_select(0, indices),
            self._joint_vel.index_select(0, indices),
            self._root_quat_w.index_select(0, indices),
        )

    def advance(self) -> bool:
        if not self._active:
            return False
        if self._frame < self.frame_count - 1:
            self._frame += 1
            return False
        self._active = False
        return True


def _finite_difference(values: torch.Tensor) -> torch.Tensor:
    velocities = torch.zeros_like(values)
    if len(values) > 1:
        velocities[:-1] = torch.diff(values, dim=0) * REFERENCE_HZ
        velocities[-1] = velocities[-2]
    return velocities
