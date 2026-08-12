from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
from mcp.types import ImageContent, TextContent
from mjlab.viewer import ViewerConfig
from tasks.catalog import TASKS
from tasks.spec import TaskSpec

from mjlab_scout.config import ScoutConfig
from mjlab_scout.runtime import ScoutRuntime
from mjlab_scout.schemas import CapturedView, ScoutView, TaskInfo
from mjlab_scout.tools import ScoutTools


class _FakeModel:
    nbody = 2
    ncam = 1
    body_geomnum = np.array([0, 1])
    body_pos = np.array([[0.0, 0.0, 0.0], [6.0, -2.0, 0.0]])
    opt = SimpleNamespace(timestep=0.002)

    def body(self, index: int):
        return SimpleNamespace(name=("world", "goal")[index])

    def camera(self, index: int):
        return SimpleNamespace(name="corridor_left")


def test_scout_catalog_exposes_dsrf_task_objective() -> None:
    assert set(TASKS) == {
        "portrait-corridors",
        "see-saw",
        "sokoban",
        "stack-steps",
    }
    assert TASKS["portrait-corridors"].objective == (
        "Stand in front of the image of the creator of Linux."
    )


def test_runtime_keeps_scene_and_renderer_on_one_thread(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    camera_configs = []
    model = _FakeModel()

    class FakeScene:
        def __init__(self, cfg, device):
            calls.append(("scene", threading.get_ident()))

        def compile(self):
            return model

        def initialize(self, mj_model, sim_model, data):
            calls.append(("initialize", threading.get_ident()))

        def reset(self):
            calls.append(("reset", threading.get_ident()))

        def write_data_to_sim(self):
            pass

        def update(self, dt):
            pass

    class FakeSimulation:
        def __init__(self, num_envs, cfg, model, device):
            self.mj_model = model
            self.model = object()
            self.data = object()
            self.expanded_fields = set()

        def forward(self):
            calls.append(("forward", threading.get_ident()))

    class FakeRenderer:
        def __init__(self, **kwargs):
            calls.append(("renderer", threading.get_ident()))
            camera_configs.append(kwargs["cfg"])

        def initialize(self):
            calls.append(("renderer.initialize", threading.get_ident()))

        def update(self, data, camera=None):
            calls.append((f"renderer.update:{camera}", threading.get_ident()))

        def render(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def close(self):
            calls.append(("renderer.close", threading.get_ident()))

    agent_camera = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",
        width=4,
        height=4,
    )
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1),
        sim=SimpleNamespace(),
        viewer=agent_camera,
    )
    definition = TaskSpec(
        name="portrait-corridors",
        objective="Reach the goal.",
        make_scene=lambda **kwargs: lambda spec: None,
    )
    monkeypatch.setattr("mjlab_scout.runtime.get_task", lambda task: definition)
    monkeypatch.setattr(
        "mjlab_scout.runtime.make_sim_env_cfg", lambda **kwargs: env_cfg
    )
    monkeypatch.setattr("mjlab_scout.runtime.Scene", FakeScene)
    monkeypatch.setattr("mjlab_scout.runtime.Simulation", FakeSimulation)
    monkeypatch.setattr("mjlab_scout.runtime.OffscreenRenderer", FakeRenderer)

    runtime = ScoutRuntime(
        ScoutConfig(
            device="cpu",
            image_width=4,
            image_height=4,
            preview_seconds=0,
        )
    )
    try:
        task = runtime.load_task("portrait-corridors")
        captured = runtime.capture_view("overview")
        runtime.capture_view("agent")
        runtime.capture_view("corridor_left")
    finally:
        runtime.close()

    assert captured.image.startswith(b"\xff\xd8")
    assert task.views == ("agent", "overview", "overhead", "corridor_left")
    assert camera_configs[0].azimuth == 45.0
    assert camera_configs[1] is agent_camera
    assert any(name == "renderer.update:corridor_left" for name, _ in calls)
    worker_threads = {thread_id for _, thread_id in calls}
    assert len(worker_threads) == 1
    assert next(iter(worker_threads)) != threading.get_ident()


def test_capture_tool_returns_text_and_native_mcp_image() -> None:
    class FakeRuntime:
        def list_tasks(self) -> tuple[TaskInfo, ...]:
            return ()

        def load_task(self, task: str) -> TaskInfo:
            raise NotImplementedError

        def capture_view(self, view: ScoutView = "overview") -> CapturedView:
            return CapturedView(
                task="portrait-corridors",
                view=view,
                width=2,
                height=1,
                image=b"jpeg-data",
            )

        def close_task(self) -> None:
            pass

    content = ScoutTools(FakeRuntime()).capture_view("overview")

    assert isinstance(content[0], TextContent)
    assert json.loads(content[0].text) == {
        "task": "portrait-corridors",
        "view": "overview",
        "width": 2,
        "height": 1,
        "mime_type": "image/jpeg",
    }
    assert isinstance(content[1], ImageContent)
    assert content[1].mimeType == "image/jpeg"
