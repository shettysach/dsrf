from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import mujoco.viewer
import torch
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from shared.messages import REFERENCE_HZ

if TYPE_CHECKING:
    from mjlab.viewer import EnvProtocol

    from sim.env import MjlabEnv
    from tracker.reference import MotionReference

from sim.reference_ghost import ReferenceGhost


class SimViewer(Protocol):
    def sync(self) -> None: ...

    def close(self) -> None: ...


class NativeSimViewer(NativeMujocoViewer):
    """Passive MJLab viewer that never owns simulation stepping."""

    def __init__(
        self,
        simulation: MjlabEnv,
        reference: MotionReference | None = None,
    ) -> None:
        super().__init__(
            cast("EnvProtocol", simulation),
            _ViewerOnlyPolicy(),
            frame_rate=float(REFERENCE_HZ),
            enable_perturbations=False,
        )
        self._reference_ghost = (
            ReferenceGhost(simulation.unwrapped, reference)
            if reference is not None
            else None
        )
        self.setup()
        self.sync()

    def sync(self) -> None:
        self.sync_env_to_viewer()

    def _update_debug_visualizers(self, viewer: mujoco.viewer.Handle) -> None:
        super()._update_debug_visualizers(viewer)
        if self._reference_ghost is None or not self._show_debug_vis:
            return

        assert self.mjm is not None
        visualizer = MujocoNativeDebugVisualizer(
            viewer.user_scn,
            self.mjm,
            self.env_idx,
            self._show_all_envs,
        )
        self._reference_ghost.draw(visualizer)


class ViserSimViewer(ViserPlayViewer):
    """Passive MJLab Viser display that never owns simulation stepping."""

    def __init__(
        self,
        simulation: MjlabEnv,
        reference: MotionReference | None = None,
    ) -> None:
        super().__init__(
            cast("EnvProtocol", simulation),
            _ViewerOnlyPolicy(),
            frame_rate=float(REFERENCE_HZ),
        )
        self._reference_ghost = (
            ReferenceGhost(simulation.unwrapped, reference)
            if reference is not None
            else None
        )
        self.setup()
        self.sync()

    def sync(self) -> None:
        self.sync_env_to_viewer()

    def _queue_debug_visualizers(self) -> None:
        super()._queue_debug_visualizers()
        if (
            self._reference_ghost is not None
            and self._scene.debug_visualization_enabled
        ):
            self._reference_ghost.draw(self._scene)


class _ViewerOnlyPolicy:
    """Sentinel policy: the passive display must never advance simulation."""

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        raise RuntimeError("The passive simulation viewer cannot drive physics")
