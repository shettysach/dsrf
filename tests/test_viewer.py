from types import SimpleNamespace
from typing import Any, cast

import pytest
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import sim.viewer as viewer_module
from sim.viewer import NativeSimViewer, ViserSimViewer


@pytest.mark.parametrize(
    ("debug_visualization_enabled", "expected"),
    [
        (True, ["mjlab", "visualizer", "ghost"]),
        (False, ["mjlab"]),
    ],
)
def test_reference_ghost_follows_mjlab_visualizers(
    monkeypatch: pytest.MonkeyPatch,
    debug_visualization_enabled: bool,
    expected: list[str],
) -> None:
    calls: list[str] = []
    debug_visualizer = object()

    monkeypatch.setattr(
        NativeMujocoViewer,
        "_update_debug_visualizers",
        lambda self, viewer: calls.append("mjlab"),
    )
    monkeypatch.setattr(
        viewer_module,
        "MujocoNativeDebugVisualizer",
        lambda *args: calls.append("visualizer") or debug_visualizer,
    )

    sonic_viewer = cast(Any, NativeSimViewer.__new__(NativeSimViewer))
    sonic_viewer._reference_ghost = SimpleNamespace(
        draw=lambda visualizer: calls.append("ghost")
    )
    sonic_viewer._show_debug_vis = debug_visualization_enabled
    sonic_viewer._show_all_envs = False
    sonic_viewer.env_idx = 0
    sonic_viewer.mjm = object()

    sonic_viewer._update_debug_visualizers(SimpleNamespace(user_scn=object()))

    assert calls == expected


def test_viser_reference_ghost_follows_mjlab_visualizers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        ViserPlayViewer,
        "_queue_debug_visualizers",
        lambda self: calls.append("mjlab"),
    )

    sonic_viewer = cast(Any, ViserSimViewer.__new__(ViserSimViewer))
    sonic_viewer._reference_ghost = SimpleNamespace(
        draw=lambda visualizer: calls.append("ghost")
    )
    sonic_viewer._scene = SimpleNamespace(debug_visualization_enabled=True)

    sonic_viewer._queue_debug_visualizers()

    assert calls == ["mjlab", "ghost"]
