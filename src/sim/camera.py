from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING

import torch
from mjlab.sensor import CameraSensor

if TYPE_CHECKING:
    from mjlab.sim import Simulation

    from tracker.state import RobotState


@dataclass(frozen=True)
class ProjectionContext:
    depth: torch.Tensor
    camera_pos_w: torch.Tensor
    camera_rotation_w: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    fovy_rad: float
    near: float
    far: float


#    MJLab currently calls ``Simulation.sense`` on every environment reset and step.
#    This project has no per-step sensor consumers, so the owned simulation instance is gated and the original bound method is called only for an explicit capture.
#    Remove this adapter when https://github.com/mujocolab/mjlab/pull/1025 lands.


class OnDemandCameraCapture:
    def __init__(
        self,
        simulation: Simulation,
        camera: CameraSensor,
    ) -> None:
        self._simulation = simulation
        self._camera = camera
        self._sense = simulation.sense
        simulation.sense = MethodType(_skip_sense, simulation)  # ty: ignore[invalid-assignment]

    def capture(
        self,
        state: RobotState,
    ) -> tuple[torch.Tensor, ProjectionContext]:
        camera_id = self._camera.camera_idx
        self._sense()

        data = self._camera.data
        assert data.rgb is not None
        assert data.depth is not None

        camera_pos_w = self._simulation.data.cam_xpos[0, camera_id]
        camera_rotation_w = self._simulation.data.cam_xmat[0, camera_id].reshape(3, 3)
        model_host = self._simulation.mj_model
        extent = float(model_host.stat.extent)
        return data.rgb[0], ProjectionContext(
            depth=data.depth[0, :, :, 0],
            camera_pos_w=camera_pos_w,
            camera_rotation_w=camera_rotation_w,
            root_pos_w=state.root_pos_w,
            root_quat_w=state.root_quat_w,
            fovy_rad=math.radians(float(model_host.cam_fovy[camera_id])),
            near=float(model_host.vis.map.znear) * extent,
            far=float(model_host.vis.map.zfar) * extent,
        )

    def close(self) -> None:
        self._simulation.sense = self._sense  # ty: ignore[invalid-assignment]


def _skip_sense(_simulation: Simulation) -> None:
    pass
