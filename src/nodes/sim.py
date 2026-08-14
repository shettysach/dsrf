from __future__ import annotations

from datetime import datetime
from typing import Optional

from dora import Node

from shared.config import SimConfig
from sim.env import MjlabEnv
from sim.renderer import SimRenderer
from sim.runtime import SimRuntime, portrait_corridor_demo_runs
from sim.sonic.policy import SonicPolicy
from sim.video import DemoVideoRecorder
from sim.viewer import NativeSimViewer, SimViewer, ViserSimViewer


def main() -> None:
    cfg = SimConfig.from_env()
    is_portrait_corridors = (
        cfg.task is not None and cfg.task.name == "portrait-corridors"
    )

    node = Node()
    simulation = MjlabEnv(
        device=cfg.device,
        task=cfg.task,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
        goal_index=cfg.goal_index,
        camera_yaw=cfg.camera_yaw,
    )
    viewer: Optional[SimViewer] = None

    try:
        with simulation.compute_context():
            policy = SonicPolicy(
                cfg.sonic_dir,
                device=cfg.device,
                cuda_stream=simulation.cuda_stream,
            )
        if cfg.viewer in {"native", "viser"}:
            reference = policy.reference if cfg.reference_ghost else None
            viewer = (
                NativeSimViewer(simulation, reference)
                if cfg.viewer == "native"
                else ViserSimViewer(simulation, reference)
            )
        renderer = SimRenderer(simulation, jpeg_quality=cfg.jpeg_quality)
        video_path = (
            cfg.demo_video_dir
            / f"goal{cfg.goal_index}_{datetime.now().strftime('%H%M')}.mp4"
            if cfg.demo_video_dir is not None
            else None
        )
        recorder = DemoVideoRecorder(video_path) if video_path is not None else None
        _log_init(node, cfg)
        SimRuntime(
            node,
            simulation,
            policy,
            renderer,
            viewer,
            recorder,
            stop_recording_at_corridor=is_portrait_corridors,
            motion_timeout_seconds=cfg.motion_timeout_seconds,
            demo_runs=(
                portrait_corridor_demo_runs(cfg.demo_runs)
                if is_portrait_corridors and recorder is not None
                else ()
            ),
        ).run()
    finally:
        if "recorder" in locals() and recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        simulation.close()


if __name__ == "__main__":
    main()


def _log_init(node: Node, cfg: SimConfig) -> None:
    node.log(
        "info",
        "Simulation initialized",
        target="dsrf.sim",
        fields={
            "event": "sim_initialized",
            "task": cfg.task.name if cfg.task is not None else "none",
            "device": cfg.device,
            "viewer": cfg.viewer,
            "reference_ghost": str(cfg.reference_ghost).lower(),
            "camera_yaw": str(cfg.camera_yaw).lower(),
        },
    )
