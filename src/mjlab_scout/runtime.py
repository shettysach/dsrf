from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, TypeVar

import imageio.v3 as iio
from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab.viewer import OffscreenRenderer, ViewerConfig
from tasks.catalog import TASKS, get_task
from tasks.spec import TaskSpec

from mjlab_scout.config import ScoutConfig
from mjlab_scout.schemas import CapturedView, ScoutView, TaskInfo
from sim.config import make_sim_env_cfg

DEFAULT_VIEWS: tuple[ScoutView, ...] = ("agent", "overview", "overhead")
JPEG_QUALITY = 85
ResultT = TypeVar("ResultT")


@dataclass
class _LoadedTask:
    name: str
    definition: TaskSpec
    scene: Scene
    sim: Simulation
    agent_camera: ViewerConfig
    views: tuple[ScoutView, ...]


class ScoutRuntime:
    """Own one MJLab scene and its renderer on a dedicated thread."""

    def __init__(self, config: ScoutConfig | None = None) -> None:
        self.config = config or ScoutConfig()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scout")
        self._loaded: _LoadedTask | None = None
        self._renderer: OffscreenRenderer | None = None
        self._renderer_view: ScoutView | None = None
        self._closed = False

    def list_tasks(self) -> tuple[TaskInfo, ...]:
        return tuple(
            TaskInfo(name=name, objective=definition.objective, views=DEFAULT_VIEWS)
            for name, definition in TASKS.items()
        )

    def load_task(self, task: str) -> TaskInfo:
        return self._submit(self._load_task, task)

    def capture_view(self, view: ScoutView = "overview") -> CapturedView:
        captured = self._submit(self._capture_view, view)
        if self.config.preview_seconds > 0:
            _spawn_preview(captured.image, captured.view, self.config.preview_seconds)
        return captured

    def close_task(self) -> None:
        self._submit(self._close_task)

    def close(self) -> None:
        if self._closed:
            return
        self._submit(self._close_task)
        self._closed = True
        self._worker.shutdown(wait=True)

    def _submit(self, fn: Callable[..., ResultT], *args: Any) -> ResultT:
        if self._closed:
            raise RuntimeError("Scout runtime is closed")
        return self._worker.submit(fn, *args).result()

    def _load_task(self, task: str) -> TaskInfo:
        definition = get_task(task)
        self._close_task()

        with redirect_stdout(sys.stderr):
            env_cfg = make_sim_env_cfg(
                image_width=self.config.image_width,
                image_height=self.config.image_height,
                task=definition,
            )
            env_cfg.scene.num_envs = 1
            scene = Scene(env_cfg.scene, device=self.config.device)
            model = scene.compile()
            sim = Simulation(
                num_envs=1,
                cfg=env_cfg.sim,
                model=model,
                device=self.config.device,
            )
            scene.initialize(sim.mj_model, sim.model, sim.data)
            scene.reset()
            scene.write_data_to_sim()
            sim.forward()
            scene.update(sim.mj_model.opt.timestep)

        self._loaded = _LoadedTask(
            name=task,
            definition=definition,
            scene=scene,
            sim=sim,
            agent_camera=env_cfg.viewer,
            views=(*DEFAULT_VIEWS, *_corridor_views(sim.mj_model)),
        )
        if self.config.preview_seconds > 0:
            overview = self._capture_view("overview")
            _spawn_preview(
                overview.image,
                overview.view,
                self.config.preview_seconds,
            )
        return TaskInfo(
            name=task,
            objective=definition.objective,
            views=self._loaded.views,
        )

    def _capture_view(self, view: ScoutView) -> CapturedView:
        loaded = self._require_loaded()
        if view not in loaded.views:
            choices = ", ".join(loaded.views)
            raise ValueError(f"Unknown view {view!r}. Available: {choices}")

        if self._renderer_view != view:
            self._close_renderer()
            self._renderer = OffscreenRenderer(
                model=loaded.sim.mj_model,
                cfg=self._camera_for(view, loaded),
                scene=loaded.scene,
                sim_model=loaded.sim.model,
                expanded_fields=loaded.sim.expanded_fields,
            )
            with redirect_stdout(sys.stderr):
                self._renderer.initialize()
            self._renderer_view = view

        assert self._renderer is not None
        with redirect_stdout(sys.stderr):
            camera_name = None if view in DEFAULT_VIEWS else view
            self._renderer.update(loaded.sim.data, camera=camera_name)
            image = _encode_jpeg(self._renderer.render())
        return CapturedView(
            task=loaded.name,
            view=view,
            width=self.config.image_width,
            height=self.config.image_height,
            image=image,
        )

    def _camera_for(self, view: ScoutView, loaded: _LoadedTask) -> ViewerConfig:
        if view not in DEFAULT_VIEWS:
            return ViewerConfig(
                width=self.config.image_width,
                height=self.config.image_height,
                max_extra_envs=0,
            )

        center, distance = _scene_frame(loaded.sim.mj_model)
        if view == "agent":
            return loaded.agent_camera
        if view == "overview":
            return ViewerConfig(
                origin_type=ViewerConfig.OriginType.WORLD,
                lookat=(center[0], center[1], 0.75),
                distance=distance,
                azimuth=0.0,
                elevation=-38.0,
                width=self.config.image_width,
                height=self.config.image_height,
                max_extra_envs=0,
            )
        return ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(center[0], center[1], 0.0),
            distance=distance,
            azimuth=90.0,
            elevation=-89.0,
            width=self.config.image_width,
            height=self.config.image_height,
            max_extra_envs=0,
        )

    def _close_task(self) -> None:
        self._close_renderer()
        self._loaded = None

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            with redirect_stdout(sys.stderr):
                self._renderer.close()
        self._renderer = None
        self._renderer_view = None

    def _require_loaded(self) -> _LoadedTask:
        if self._loaded is None:
            raise RuntimeError("Load a task before inspecting the scene")
        return self._loaded


def _encode_jpeg(image: Any) -> bytes:
    buffer = BytesIO()
    iio.imwrite(buffer, image, extension=".jpg", quality=JPEG_QUALITY)
    return buffer.getvalue()


def _spawn_preview(image: bytes, view: ScoutView, duration: float) -> None:
    """Show a rendered frame without blocking the Scout's render thread."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("Scout overview preview skipped: no graphical display", file=sys.stderr)
        return

    fd, path = tempfile.mkstemp(prefix="mjlab-scout-overview-", suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as image_file:
            image_file.write(image)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mjlab_scout.preview",
                "--duration",
                str(duration),
                "--view",
                view,
                path,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        print(f"Scout overview preview failed: {exc}", file=sys.stderr)
        return

    threading.Thread(target=process.wait, name="scout-preview", daemon=True).start()


def _scene_frame(model: Any) -> tuple[tuple[float, float], float]:
    positions = []
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name
        if not name or "/" in name or name == "terrain":
            continue
        if int(model.body_geomnum[body_id]) > 0:
            x, y, _ = model.body_pos[body_id]
            positions.append((float(x), float(y)))
    if not positions:
        return (0.0, 0.0), 8.0
    xs = [position[0] for position in positions]
    ys = [position[1] for position in positions]
    center = ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5)
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return center, max(8.0, span * 1.35)


def _corridor_views(model: Any) -> tuple[ScoutView, ...]:
    return tuple(
        name
        for camera_id in range(model.ncam)
        if (name := model.camera(camera_id).name) and name.startswith("corridor_")
    )
