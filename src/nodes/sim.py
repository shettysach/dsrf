from __future__ import annotations

from typing import Optional

from dora import Node

from motion_gen.factory import create_motion_generator
from shared.config import MotionGenConfig, SimConfig
from sim.env import MjlabEnv
from sim.renderer import SimRenderer
from sim.runtime import SimRuntime
from sim.video import DemoVideoRecorder
from sim.viewer import NativeSimViewer, SimViewer, ViserSimViewer
from tracker.sonic import SonicTracker


def main() -> None:
    cfg = SimConfig.from_env()
    motion_cfg = MotionGenConfig.from_env()

    node = Node()
    simulation = MjlabEnv(
        device=cfg.device,
        task=cfg.task,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
    )
    viewer: Optional[SimViewer] = None
    recorder: DemoVideoRecorder | None = None

    try:
        with simulation.compute_context():
            generator = create_motion_generator(
                motion_cfg,
                cuda_stream=simulation.cuda_stream,
            )
            tracker = SonicTracker(
                cfg.sonic_dir,
                device=cfg.device,
                cuda_stream=simulation.cuda_stream,
            )
        if cfg.viewer in {"native", "viser"}:
            reference = tracker.reference if cfg.reference_ghost else None
            viewer = (
                NativeSimViewer(simulation.mjlab_env, reference)
                if cfg.viewer == "native"
                else ViserSimViewer(simulation.mjlab_env, reference)
            )
        renderer = SimRenderer(simulation, jpeg_quality=cfg.jpeg_quality)
        recorder = (
            DemoVideoRecorder(cfg.demo_video_path)
            if cfg.demo_video_path is not None
            else None
        )
        _log_init(node, cfg)
        SimRuntime(
            node,
            simulation,
            generator,
            tracker,
            renderer,
            viewer,
            recorder,
            stop_on_stand=cfg.stop_on_stand,
            max_completed_commands=cfg.demo_max_commands,
            timeout_seconds=cfg.demo_timeout_seconds,
            publish_observations=cfg.publish_observations,
        ).run()
    finally:
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
        simulation.close()


def _log_init(node: Node, cfg: SimConfig) -> None:
    node.log(
        "info",
        "Simulation initialized",
        target="dsrf.sim",
        fields={
            "event": "sim_initialized",
            "task": cfg.task.name if cfg.task is not None else "none",
            "device": cfg.device,
            "tracker": "sonic",
            "viewer": cfg.viewer,
            "reference_ghost": str(cfg.reference_ghost).lower(),
            "demo_video_path": str(cfg.demo_video_path) if cfg.demo_video_path else "",
        },
    )


if __name__ == "__main__":
    main()
