from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from dora import Node

from shared.arrow import (
    grounding_request_from_arrow,
    grounding_result_to_arrow,
    motion_from_arrow,
    observation_to_arrow,
    pipeline_error_to_arrow,
)
from shared.messages import (
    REFERENCE_HZ,
    EndEffectorTarget,
    GroundingResult,
    PipelineError,
    VisualObservation,
)
from sim.camera import ProjectionContext
from sim.env import MjlabEnv
from sim.grounding import resolve_end_effector, resolve_waypoint
from sim.renderer import SimRenderer
from sim.viewer import SimViewer
from tracker.sonic import SonicTracker


@dataclass(frozen=True)
class ExecutionStats:
    frames: int
    elapsed_ms: float
    overrun_steps: int


class SimRuntime:
    def __init__(
        self,
        node: Node,
        simulation: MjlabEnv,
        tracker: SonicTracker,
        renderer: SimRenderer,
        viewer: SimViewer | None = None,
    ) -> None:
        self.node = node
        self.simulation = simulation
        self.tracker = tracker
        self.renderer = renderer
        self.viewer = viewer
        self.observation_id = 0
        self._projection_cache: ProjectionContext | None = None
        self._observation_published_at: float | None = None
        expected_step_dt = 1.0 / REFERENCE_HZ
        if abs(self.simulation.step_dt - expected_step_dt) >= 1.0e-9:
            raise ValueError(
                f"Simulation step_dt must be {expected_step_dt}, "
                f"got {self.simulation.step_dt}"
            )

    def run(self) -> None:
        render_ms, jpeg_size = self._publish_observation(completed_command=None)
        self.node.log(
            "info",
            f"[OBS 0] initial observation: render_ms={render_ms:.1f} "
            f"jpeg_kb={jpeg_size / 1024.0:.1f} waiting=motion",
            target="dsrf.sim",
            fields={
                "event": "initial_observation",
                "observation_id": "0",
                "render_ms": f"{render_ms:.1f}",
                "jpeg_kb": f"{jpeg_size / 1024.0:.1f}",
            },
        )
        for event in self.node:
            if event["type"] == "STOP":
                return
            if event["type"] != "INPUT":
                continue
            if event["id"] == "motion":
                self._accept_motion(event)
            elif event["id"] == "grounding_request":
                self._accept_grounding_request(event)

    def _accept_grounding_request(self, event: dict[str, Any]) -> None:
        metadata = dict(event.get("metadata") or {})
        try:
            request = grounding_request_from_arrow(event["value"], metadata)
            if request.observation_id != self.observation_id:
                raise ValueError(
                    f"Expected grounding request for observation {self.observation_id}, "
                    f"got {request.observation_id}"
                )
            if self._projection_cache is None:
                raise ValueError(
                    "No RGB-D capture is available for the current observation"
                )
            resolved = tuple(
                resolve_waypoint(waypoint, self._projection_cache)
                for waypoint in request.waypoints_2d
            )
            resolved_end_effectors = tuple(
                resolve_end_effector(
                    selection.name, selection.target_2d, self._projection_cache
                )
                for selection in request.end_effectors_2d
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._report_error(str(exc), source="grounding")
            return

        self.node.log(
            "info",
            f"[OBS {self.observation_id}] constraints grounded: "
            f"waypoints={[waypoint.target_xy for waypoint in resolved]} "
            f"end_effectors={[(target.name, target.target_xyz) for target in resolved_end_effectors]}",
            target="dsrf.sim.grounding",
            fields={
                "event": "constraints_grounded",
                "observation_id": str(self.observation_id),
            },
        )
        result = GroundingResult(
            self.observation_id,
            tuple(waypoint.target_xy for waypoint in resolved),
            tuple(
                EndEffectorTarget(target.name, target.target_xyz)
                for target in resolved_end_effectors
            ),
        )
        data, result_metadata = grounding_result_to_arrow(result)
        self.node.send_output("grounding_result", data, metadata=result_metadata)

    def _accept_motion(self, event: dict[str, Any]) -> None:
        received_at = time.perf_counter()
        metadata = dict(event.get("metadata") or {})
        try:
            chunk = motion_from_arrow(event["value"], metadata)
        except (KeyError, TypeError, ValueError) as exc:
            self._report_error(str(exc))
            return
        if chunk.observation_id != self.observation_id:
            self._report_error(
                f"Expected motion for observation {self.observation_id}, got "
                f"{chunk.observation_id}"
            )
            return

        self._projection_cache = None

        with self.simulation.compute_context():
            state = self.simulation.robot_state()
            try:
                self.tracker.load_motion(chunk, state)
            except ValueError as exc:
                self._report_error(str(exc))
                return

        published_at = self._observation_published_at
        pause_ms = (
            (received_at - published_at) * 1000.0 if published_at is not None else 0.0
        )
        stats = self._execute()
        completed_observation_id = self.observation_id
        self.observation_id += 1
        render_ms, jpeg_size = self._publish_observation(
            completed_command=chunk.command
        )
        target_ms = stats.frames * self.simulation.step_dt * 1000.0
        realtime = target_ms / stats.elapsed_ms if stats.elapsed_ms > 0.0 else 0.0
        self.node.log(
            "info",
            f"[OBS {completed_observation_id}->{self.observation_id}] motion complete: "
            f"command={chunk.command!r} pause_ms={pause_ms:.1f} "
            f"frames={stats.frames} target_ms={target_ms:.1f} "
            f"exec_ms={stats.elapsed_ms:.1f} realtime={realtime:.3f} "
            f"render_ms={render_ms:.1f}",
            target="dsrf.sim",
            fields={
                "event": "motion_complete",
                "observation_id": str(completed_observation_id),
                "next_observation_id": str(self.observation_id),
                "command": chunk.command,
                "pause_ms": f"{pause_ms:.1f}",
                "frames": str(stats.frames),
                "target_ms": f"{target_ms:.1f}",
                "exec_ms": f"{stats.elapsed_ms:.1f}",
                "realtime": f"{realtime:.3f}",
                "overrun_steps": str(stats.overrun_steps),
                "render_ms": f"{render_ms:.1f}",
                "jpeg_kb": f"{jpeg_size / 1024.0:.1f}",
            },
        )

    def _execute(self) -> ExecutionStats:
        started_at = time.perf_counter()
        next_step = time.perf_counter()
        frames = 0
        overrun_steps = 0
        with torch.no_grad():
            while True:
                delay = next_step - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)

                with self.simulation.compute_context():
                    state = self.simulation.robot_state()
                    action, completed = self.tracker.act(state)
                self.simulation.step(action)
                if self.viewer is not None:
                    self.viewer.sync()
                frames += 1

                # Completion is detected while producing the last reference
                # action. Capture only after that action's physics step.
                if completed:
                    return ExecutionStats(
                        frames=frames,
                        elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                        overrun_steps=overrun_steps,
                    )

                next_step += self.simulation.step_dt
                now = time.perf_counter()
                if next_step < now:
                    # Do not execute burst catch-up steps after an overrun.
                    overrun_steps += 1
                    next_step = now

    def _publish_observation(
        self, *, completed_command: str | None
    ) -> tuple[float, int]:
        render_started_at = time.perf_counter()
        jpeg, projection = self.renderer.capture_rgbd()
        render_ms = (time.perf_counter() - render_started_at) * 1000.0
        observation = VisualObservation(
            observation_id=self.observation_id,
            completed_command=completed_command,
            jpeg=jpeg,
        )
        data, metadata = observation_to_arrow(observation)
        self.node.send_output("observation", data, metadata=metadata)
        self._projection_cache = projection
        self._observation_published_at = time.perf_counter()
        return render_ms, len(jpeg)

    def _report_error(self, detail: str, *, source: str = "sim") -> None:
        self.node.log(
            "error",
            f"[OBS {self.observation_id}] {source} error: {detail}",
            target="dsrf.sim",
            fields={
                "event": "pipeline_error",
                "observation_id": str(self.observation_id),
                "source": source,
                "detail": detail,
            },
        )
        error = PipelineError(source, self.observation_id, detail)
        self.node.send_output("error", pipeline_error_to_arrow(error))
