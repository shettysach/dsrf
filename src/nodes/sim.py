from __future__ import annotations

from typing import Optional

from dora import Node

from controller import Controller
from controller.direct import DirectController
from controller.sonic import SonicController
from shared.config import DirectConfig, SimConfig, SonicConfig
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
        control_mode=(
            "pd" if isinstance(cfg.controller, DirectConfig) else "position"
        ),
    )
    viewer: Optional[SimViewer] = None

    try:
        with simulation.compute_context():
            controller = _create_controller(cfg, simulation)
        if cfg.viewer in {"native", "viser"}:
            reference = (
                getattr(controller, "reference", None) if cfg.reference_ghost else None
            )
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


def _log_init(node: Node, cfg: SimConfig) -> None:
    node.log(
        "info",
        "Simulation initialized",
        target="dsrf.sim",
        fields={
            "event": "sim_initialized",
            "task": cfg.task.name if cfg.task is not None else "none",
            "device": cfg.device,
            "controller": type(cfg.controller).__name__,
            "viewer": cfg.viewer,
            "reference_ghost": str(cfg.reference_ghost).lower(),
        },
    )


def _create_controller(cfg: SimConfig, simulation: MjlabEnv) -> Controller:
    match cfg.controller:
        case SonicConfig():
            return SonicController(
                cfg.controller.sonic_dir,
                simulation.command_transform,
                device=cfg.device,
                cuda_stream=simulation.cuda_stream,
            )
        case DirectConfig():
            return DirectController(
                cfg.controller,
                robot_mass=simulation.robot_mass,
                gravity_magnitude=simulation.gravity_magnitude,
                mj_model=simulation.mj_model,
                robot_indexing=simulation.robot_indexing,
                device=cfg.device,
            )


if __name__ == "__main__":
    main()
