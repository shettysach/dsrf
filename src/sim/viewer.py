from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any, Protocol, cast
from unittest.mock import patch

import mujoco.viewer
import torch
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from shared.messages import SONIC_FPS

if TYPE_CHECKING:
    from mjlab.viewer import EnvProtocol

    from sim.env import MjlabEnv
    from sim.sonic.policy import MotionReference

from sim.reference_ghost import ReferenceGhost


class SimViewer(Protocol):
    def sync(self) -> None: ...

    def close(self) -> None: ...

    def set_vlm_thinking(self, observation_id: int) -> None: ...

    def set_vlm_result(
        self, observation_id: int, reasoning: str | None, command: str
    ) -> None: ...


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
            frame_rate=float(SONIC_FPS),
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

    def set_vlm_thinking(self, observation_id: int) -> None:
        del observation_id

    def set_vlm_result(
        self, observation_id: int, reasoning: str | None, command: str
    ) -> None:
        del observation_id, reasoning, command

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
            frame_rate=float(SONIC_FPS),
        )
        self._reference_ghost = (
            ReferenceGhost(simulation.unwrapped, reference)
            if reference is not None
            else None
        )
        self.setup()
        self.sync()

    def set_vlm_thinking(self, observation_id: int) -> None:
        self._vlm_panel.content = _vlm_panel_html(
            observation_id,
            reasoning="Thinking…",
            command="",
            thinking=True,
        )

    def set_vlm_result(
        self, observation_id: int, reasoning: str | None, command: str
    ) -> None:
        self._vlm_panel.content = _vlm_panel_html(
            observation_id,
            reasoning=reasoning or "(No reasoning returned)",
            command=command,
        )

    def setup(self) -> None:
        # Reuse MJLab's scene/control initialization, then discard its GUI
        # because the demo sidebar is intentionally VLM-only.
        tab_groups: list[Any] = []
        add_tab_group = self._server.gui.add_tab_group

        def capture_tab_group(*args: Any, **kwargs: Any) -> Any:
            group = add_tab_group(*args, **kwargs)
            tab_groups.append(group)
            return group

        with patch.object(self._server.gui, "add_tab_group", capture_tab_group):
            super().setup()
        assert len(tab_groups) == 1
        tab_group = tab_groups[0]
        # Keep MJLab's GUI handles alive because its sync path updates them,
        # but detach their tabs from the visible group.
        tab_group._tab_labels = ()
        tab_group._tab_icons_html = ()
        tab_group._tab_handles = []
        tab_group._tab_container_ids = ()

        with tab_group.add_tab("VLM"):
            self._vlm_panel = self._server.gui.add_html(
                _vlm_panel_html(
                    -1,
                    reasoning="Waiting for observation…",
                    command="",
                    thinking=True,
                )
            )

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


def _vlm_panel_html(
    observation_id: int,
    *,
    reasoning: str,
    command: str,
    thinking: bool = False,
) -> str:
    observation = "—" if observation_id < 0 else str(observation_id)
    reasoning_html = html.escape(reasoning)
    command_html = html.escape(command)
    decision = (
        ""
        if thinking
        else (
            '<div style="font-size:10px;font-weight:700;color:#94a3b8;margin-top:10px;">'
            "Command</div>"
            f'<div style="font-size:12px;font-weight:600;margin-top:2px;'
            f'overflow-wrap:anywhere;max-width:100%;">'
            f"{command_html}</div>"
        )
    )
    return (
        '<div style="font-family:system-ui,sans-serif;font-size:11px;'
        "line-height:1.35;padding:2px 0;width:100%;max-width:100%;"
        'box-sizing:border-box;overflow:hidden;">'
        f'<div style="font-size:11px;margin-bottom:8px;">Observation #{observation}</div>'
        '<div style="font-size:12px;font-weight:700;margin-bottom:6px;">'
        "VLM output</div>"
        '<div style="font-size:10px;font-weight:700;color:#94a3b8;">'
        "Reasoning</div>"
        '<div style="font-size:11px;line-height:1.35;max-height:180px;'
        "max-width:100%;overflow-y:auto;overflow-wrap:anywhere;"
        f'white-space:pre-wrap;margin-top:2px;">{reasoning_html}</div>'
        f"{decision}"
        "</div>"
    )
