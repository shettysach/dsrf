from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING

import torch
from mjlab.sensor import CameraSensor
from mjlab.utils.lab_api.math import quat_from_matrix
from tasks import ObservationCameraSpec

if TYPE_CHECKING:
    from mjlab.sensor import SensorContext
    from mjlab.sim import Simulation

    from controller import RobotState


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


@dataclass(frozen=True)
class RgbdFrame:
    rgb: torch.Tensor
    projection: ProjectionContext


class OnDemandCameraCapture:
    """Own sparse MJLab camera sensing until upstream supports render intervals.

    MJLab currently calls ``Simulation.sense`` on every environment reset and step.
    This project has no per-step sensor consumers, so the owned simulation instance is
    gated and the original bound method is called only for an explicit capture.

    Remove this adapter when https://github.com/mujocolab/mjlab/pull/1025 lands.
    """

    def __init__(
        self,
        simulation: Simulation,
        camera: CameraSensor,
        sensor_context: SensorContext,
        spec: ObservationCameraSpec,
    ) -> None:
        if not sensor_context.has_cameras:
            raise RuntimeError("On-demand capture requires an MJLab camera sensor")
        if sensor_context.has_raycasts:
            raise RuntimeError("On-demand capture cannot gate raycast sensing")

        original_sense = simulation.sense
        if inspect.signature(original_sense).parameters:
            raise RuntimeError(
                "MJLab sense() has changed; remove OnDemandCameraCapture and use "
                "the upstream on-demand rendering API"
            )

        self._simulation = simulation
        self._camera = camera
        self._spec = spec
        self._sense = original_sense
        self._closed = False
        simulation.sense = MethodType(_skip_sense, simulation)  # ty: ignore[invalid-assignment]

    def capture(self, state: RobotState) -> RgbdFrame:
        target = torch.as_tensor(
            self._spec.lookat,
            dtype=state.root_pos_w.dtype,
            device=state.root_pos_w.device,
        )
        if self._spec.origin == "robot":
            target = target + state.root_pos_w

        camera_pos, camera_rotation = orbit_camera_pose(
            target,
            distance=self._spec.distance,
            elevation_deg=self._spec.elevation,
            azimuth_deg=self._spec.azimuth,
        )
        camera_id = self._camera.camera_idx
        model = self._simulation.model
        _write_camera_field(model.cam_pos, camera_id, camera_pos)
        _write_camera_field(
            model.cam_quat,
            camera_id,
            quat_from_matrix(camera_rotation),
        )

        self._simulation.forward()
        self._sense()
        data = self._camera.data
        if data.rgb is None or data.depth is None:
            raise RuntimeError("Observation camera must provide RGB and depth")

        camera_pos_w = self._simulation.data.cam_xpos[0, camera_id]
        camera_rotation_w = self._simulation.data.cam_xmat[0, camera_id].reshape(3, 3)
        model_host = self._simulation.mj_model
        extent = float(model_host.stat.extent)
        return RgbdFrame(
            rgb=data.rgb[0],
            projection=ProjectionContext(
                depth=data.depth[0, :, :, 0],
                camera_pos_w=camera_pos_w,
                camera_rotation_w=camera_rotation_w,
                root_pos_w=state.root_pos_w,
                root_quat_w=state.root_quat_w,
                fovy_rad=math.radians(self._spec.fovy),
                near=float(model_host.vis.map.znear) * extent,
                far=float(model_host.vis.map.zfar) * extent,
            ),
        )

    def close(self) -> None:
        if not self._closed:
            self._simulation.sense = self._sense  # ty: ignore[invalid-assignment]
            self._closed = True


def orbit_camera_pose(
    target_w: torch.Tensor,
    *,
    distance: float,
    elevation_deg: float,
    azimuth_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a MuJoCo camera pose matching its free-camera orbit convention."""

    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    offset = target_w.new_tensor(
        (
            -math.cos(elevation) * math.cos(azimuth),
            -math.cos(elevation) * math.sin(azimuth),
            -math.sin(elevation),
        )
    )
    position = target_w + distance * offset
    forward = torch.nn.functional.normalize(target_w - position, dim=0)
    world_up = target_w.new_tensor((0.0, 0.0, 1.0))
    right = torch.nn.functional.normalize(torch.cross(forward, world_up, dim=0), dim=0)
    up = torch.cross(right, forward, dim=0)
    rotation = torch.stack((right, up, -forward), dim=1)
    return position, rotation


def _write_camera_field(
    field: torch.Tensor,
    camera_id: int,
    value: torch.Tensor,
) -> None:
    if field.ndim == value.ndim + 1:
        field[camera_id].copy_(value)
    else:
        field[:, camera_id].copy_(value)


def _skip_sense(_simulation: Simulation) -> None:
    pass
