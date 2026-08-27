from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
import yaml
from dora import Node

from motion_gen.generator import MotionGenerator
from motion_gen.resample import resample_qpos
from shared.arrow import (
    agent_command_from_arrow,
    grounding_request_from_arrow,
    grounding_result_to_arrow,
    observation_to_arrow,
    pipeline_error_to_arrow,
)
from shared.messages import (
    REFERENCE_HZ,
    AgentCommand,
    EndEffectorTarget,
    GroundingResult,
    PipelineError,
    VisualObservation,
)
from sim.camera import ProjectionContext
from sim.env import MjlabEnv
from sim.grounding import resolve_end_effector, resolve_waypoint
from sim.renderer import SimRenderer
from sim.video import DemoVideoRecorder, DemoVlmState
from sim.viewer import SimViewer
from sim.virtual_force import VirtualForce, VirtualForceResult
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
        generator: MotionGenerator,
        tracker: SonicTracker,
        renderer: SimRenderer,
        viewer: SimViewer | None = None,
        recorder: DemoVideoRecorder | None = None,
        stop_on_stand: bool = False,
        max_completed_commands: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.node = node
        self.simulation = simulation
        self.generator = generator
        self.tracker = tracker
        self.renderer = renderer
        self.viewer = viewer
        self.recorder = recorder
        self.stop_on_stand = stop_on_stand
        self.max_completed_commands = max_completed_commands
        self.timeout_seconds = timeout_seconds
        self.completed_commands = 0
        self.demo_vlm_state = DemoVlmState()
        self.observation_id = 0
        self._projection_cache: ProjectionContext | None = None
        self._observation_published_at: float | None = None
        self._stop_requested = False
        task = getattr(simulation, "task", None)
        self.virtual_force = (
            VirtualForce(
                task.virtual_force_objects,
                dt=simulation.step_dt,
                device=simulation.device,
                magnitude=task.virtual_force_magnitude,
                maximum=task.virtual_force_max,
            )
            if task is not None and task.virtual_force_objects
            else None
        )

    def run(self) -> None:
        timer = self._start_timeout_timer()
        try:
            render_ms, jpeg_size = self._publish_observation(completed_command=None)
            self.node.log(
                "info",
                f"[OBS 0] initial observation: render_ms={render_ms:.1f} "
                f"jpeg_kb={jpeg_size / 1024.0:.1f} waiting=command",
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
                if event["id"] == "command":
                    self._accept_command(event)
                    if self._stop_requested:
                        return
                elif event["id"] == "grounding_request":
                    self._accept_grounding_request(event)
        finally:
            if timer is not None:
                timer.cancel()

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

    def _accept_command(self, event: dict[str, Any]) -> None:
        received_at = time.perf_counter()
        metadata = dict(event.get("metadata") or {})
        try:
            command = agent_command_from_arrow(event["value"], metadata)
        except (KeyError, TypeError, ValueError) as exc:
            self._report_error(str(exc), source="motion-gen")
            return
        if command.observation_id != self.observation_id:
            self._report_error(
                f"Expected command for observation {self.observation_id}, got "
                f"{command.observation_id}"
            )
            return

        self.demo_vlm_state = DemoVlmState(
            observation_id=command.observation_id,
            reasoning=command.reasoning or "",
            command=command.text,
        )

        self._projection_cache = None
        generation_started_at = time.perf_counter()
        try:
            with self.simulation.compute_context():
                source_qpos = self.generator.generate(command)
                qpos = resample_qpos(source_qpos, source_fps=self.generator.fps)
                state = self.simulation.robot_state()
                self.tracker.load_motion(qpos, state)
                if self.virtual_force is not None:
                    self.virtual_force.load_motion(qpos, state)
        except ValueError as exc:
            self._report_error(str(exc), source="motion-gen")
            return
        except Exception as exc:
            self._log_generation_error(command, generation_started_at, exc)
            raise

        self._log_motion_generated(
            command,
            source_qpos,
            qpos,
            plan_ms=(time.perf_counter() - generation_started_at) * 1000.0,
        )

        published_at = self._observation_published_at
        pause_ms = (
            (received_at - published_at) * 1000.0 if published_at is not None else 0.0
        )
        stats = self._execute()
        completed_observation_id = self.observation_id
        self.completed_commands += 1
        if self.stop_on_stand and command.terminal:
            # Do not publish a fresh observation after the terminal command: that
            # would make the agent issue one more VLM request into a closing node.
            self._stop_requested = True
            self.node.log(
                "info",
                f"[OBS {completed_observation_id}] terminal stand completed",
                target="dsrf.sim",
                fields={
                    "event": "terminal_stand_completed",
                    "observation_id": str(completed_observation_id),
                },
            )
            return
        if (
            self.max_completed_commands is not None
            and self.completed_commands >= self.max_completed_commands
        ):
            self._stop_requested = True
            self.node.log(
                "info",
                f"Completed-motion limit reached: {self.completed_commands}",
                target="dsrf.sim",
                fields={
                    "event": "demo_max_commands_reached",
                    "completed_commands": str(self.completed_commands),
                },
            )
            return
        self.observation_id += 1
        render_ms, jpeg_size = self._publish_observation(completed_command=command.text)
        target_ms = stats.frames * self.simulation.step_dt * 1000.0
        realtime = target_ms / stats.elapsed_ms if stats.elapsed_ms > 0.0 else 0.0
        self.node.log(
            "info",
            f"[OBS {completed_observation_id}->{self.observation_id}] motion complete: "
            f"command={command.text!r} pause_ms={pause_ms:.1f} "
            f"frames={stats.frames} target_ms={target_ms:.1f} "
            f"exec_ms={stats.elapsed_ms:.1f} realtime={realtime:.3f} "
            f"render_ms={render_ms:.1f}",
            target="dsrf.sim",
            fields={
                "event": "motion_complete",
                "observation_id": str(completed_observation_id),
                "next_observation_id": str(self.observation_id),
                "command": command.text,
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
                    contacts = (
                        self.simulation.hand_object_contacts(
                            self.virtual_force.object_names
                        )
                        if self.virtual_force is not None
                        else set()
                    )
                    force_result = (
                        self.virtual_force.compute(
                            self.tracker.reference.frame_index,
                            contacts,
                        )
                        if self.virtual_force is not None
                        else None
                    )
                    action, completed = self.tracker.act(state)
                if force_result is not None:
                    self._log_hand_contacts(force_result)
                if force_result is None:
                    self.simulation.step(action)
                else:
                    self.simulation.step(action, external_forces=force_result.forces)
                if self.viewer is not None:
                    self.viewer.sync()
                if self.recorder is not None:
                    self.recorder.write_frame(
                        self.renderer.capture_demo_rgb(), self.demo_vlm_state
                    )
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

    def _log_hand_contacts(self, result: VirtualForceResult) -> None:
        for hand, object_name in result.started_contacts:
            self.node.log(
                "info",
                f"{hand} contacted {object_name}",
                target="dsrf.sim.virtual_force",
            )
        for hand, object_name in result.ended_contacts:
            self.node.log(
                "info",
                f"{hand} left {object_name}",
                target="dsrf.sim.virtual_force",
            )

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

    def _log_motion_generated(
        self,
        command: AgentCommand,
        source_qpos: torch.Tensor,
        qpos: torch.Tensor,
        *,
        plan_ms: float,
    ) -> None:
        duration_s = len(qpos) / REFERENCE_HZ
        self.node.log(
            "info",
            f"[OBS {command.observation_id}] motion generated: "
            f"command={command.text!r} frames={len(qpos)} "
            f"duration_s={duration_s:.2f} plan_ms={plan_ms:.1f}",
            target="dsrf.motion_gen",
            fields={
                "event": "motion_generated",
                "observation_id": str(command.observation_id),
                "command": command.text,
                "plan_ms": f"{plan_ms:.1f}",
                "source_frames": str(len(source_qpos)),
                "output_frames": str(len(qpos)),
                "duration_s": f"{duration_s:.2f}",
            },
        )

    def _log_generation_error(
        self,
        command: AgentCommand,
        started_at: float,
        error: Exception,
    ) -> None:
        plan_ms = (time.perf_counter() - started_at) * 1000.0
        detail = f"{type(error).__name__}: {error}"
        self.node.log(
            "error",
            f"[OBS {command.observation_id}] motion-gen error: {detail}",
            target="dsrf.motion_gen",
            fields={
                "event": "motion_generation_error",
                "observation_id": str(command.observation_id),
                "command": command.text,
                "plan_ms": f"{plan_ms:.1f}",
                "detail": detail,
            },
        )

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

    def _start_timeout_timer(self) -> threading.Timer | None:
        if self.timeout_seconds is None:
            return None
        timer = threading.Timer(self.timeout_seconds, self._request_dataflow_stop)
        timer.daemon = True
        timer.start()
        return timer

    def _request_dataflow_stop(self) -> None:
        command = ["dora", "stop"]
        dataflow_id = _current_dataflow_id()
        if dataflow_id is not None:
            command.append(dataflow_id)
        command.extend(("--grace-duration", "10s"))
        try:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self.node.log(
                "error",
                f"Failed to stop timed-out dataflow: {exc}",
                target="dsrf.sim",
            )


def _current_dataflow_id() -> str | None:
    raw_config = os.environ.get("DORA_NODE_CONFIG")
    if raw_config is None:
        return None
    try:
        config = yaml.safe_load(raw_config)
    except yaml.YAMLError:
        return None
    if not isinstance(config, dict):
        return None
    value = config.get("dataflow_id")
    return str(value) if value else None
