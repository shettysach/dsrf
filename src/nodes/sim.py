from __future__ import annotations

from typing import Optional

from dora import Node

from shared.config import SimConfig, SonicConfig
from sim.controller import Controller
from sim.controller.sonic import SonicController
from sim.env import MjlabEnv
from sim.renderer import SimRenderer
from sim.runtime import SimRuntime
from sim.viewer import NativeSimViewer, SimViewer, ViserSimViewer


def main() -> None:
    cfg = SimConfig.from_env()

    node = Node()
    simulation = MjlabEnv(
        device=cfg.device,
        task=cfg.task,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
    )
    viewer: Optional[SimViewer] = None

    try:
        with simulation.compute_context():
            controller = _create_controller(cfg, simulation)
        if cfg.viewer in {"native", "viser"}:
            reference = getattr(controller, "reference", None) if cfg.reference_ghost else None
            viewer = (
                NativeSimViewer(simulation, reference)
                if cfg.viewer == "native"
                else ViserSimViewer(simulation, reference)
            )
        renderer = SimRenderer(simulation, jpeg_quality=cfg.jpeg_quality)
        _log_init(node, cfg)
        SimRuntime(
            node,
            simulation,
            controller,
            renderer,
            viewer,
        ).run()
    finally:
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
        },
    )


def _create_controller(cfg: SimConfig, simulation: MjlabEnv) -> Controller:
    match cfg.controller:
        case SonicConfig():
            return SonicController(
                cfg.controller.sonic_dir,
                device=cfg.device,
                cuda_stream=simulation.cuda_stream,
            )
