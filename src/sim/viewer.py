from __future__ import annotations

import socket
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import mujoco.viewer
import torch
from mjlab.viewer import EnvProtocol, NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from shared.messages import REFERENCE_HZ

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from tracker.reference import MotionReference

from sim.reference_ghost import ReferenceGhost


class SimViewer(Protocol):
    def sync(self) -> None: ...

    def close(self) -> None: ...


class NativeSimViewer(NativeMujocoViewer):
    """Passive MJLab viewer that never owns simulation stepping."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        reference: MotionReference | None = None,
        *,
        keyboard_socket: Path | None = None,
    ) -> None:
        self._keyboard_socket = keyboard_socket
        super().__init__(
            cast(EnvProtocol, env),
            _ViewerOnlyPolicy(),
            frame_rate=float(REFERENCE_HZ),
            enable_perturbations=False,
            key_callback=self._send_keyboard_key
            if keyboard_socket is not None
            else None,
        )
        self._reference_ghost = (
            ReferenceGhost(env, reference) if reference is not None else None
        )
        self.setup()
        self.sync()

    def sync(self) -> None:
        self.sync_env_to_viewer()

    def _safe_key_callback(self, key: int) -> None:
        if self._keyboard_socket is not None and _keyboard_message(key) is not None:
            self._send_keyboard_key(key)
            return
        super()._safe_key_callback(key)

    def _send_keyboard_key(self, key: int) -> None:
        assert self._keyboard_socket is not None
        message = _keyboard_message(key)
        if message is None:
            return
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            try:
                sender.sendto(message, str(self._keyboard_socket))
            except OSError:
                return

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
        env: ManagerBasedRlEnv,
        reference: MotionReference | None = None,
    ) -> None:
        super().__init__(
            cast(EnvProtocol, env),
            _ViewerOnlyPolicy(),
            frame_rate=float(REFERENCE_HZ),
        )
        self._reference_ghost = (
            ReferenceGhost(env, reference) if reference is not None else None
        )
        self.setup()
        self.sync()

    def sync(self) -> None:
        # The runtime calls every viewer once per physics frame so the native
        # viewer stays live. Viser renders in the browser, however, and its
        # scene serialization is wasted until a browser client exists.
        if not self._server.get_clients():
            return
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


def _keyboard_message(key: int) -> bytes | None:
    from mjlab.viewer.native.keys import KEY_DOWN, KEY_F, KEY_LEFT, KEY_RIGHT, KEY_UP

    return {
        KEY_UP: b"up",
        KEY_DOWN: b"down",
        KEY_LEFT: b"left",
        KEY_RIGHT: b"right",
        KEY_F: b"finish",
    }.get(key)
