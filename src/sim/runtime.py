from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml
from dora import Node
from tasks.spec import TaskStepHook

from shared.arrow import (
    grounding_request_from_arrow,
    grounding_result_to_arrow,
    motion_from_arrow,
    observation_to_arrow,
    pipeline_error_to_arrow,
)
from shared.messages import (
    SONIC_FPS,
    EndEffectorTarget,
    GroundingResult,
    MotionChunk,
    PipelineError,
    ProjectionContext,
    VisualObservation,
)
from sim.env import MjlabEnv
from sim.grounding import resolve_end_effector, resolve_waypoint
from sim.renderer import SimRenderer
from sim.sonic.policy import SonicPolicy
from sim.video import DemoVideoRecorder, DemoVlmState
from sim.viewer import SimViewer

CONTROL_PERIOD = 1.0 / SONIC_FPS
PORTRAIT_CORRIDOR_APPROACH_X = 1.0
MAX_REPEATED_COMMANDS = 10

PORTRAIT_CORRIDOR_RUNS = (
    ("Center far back", (-1.75, 0.0)),
    ("Center back", (-1.25, 0.0)),
    ("Center left back", (-1.25, 0.6)),
    ("Center right back", (-1.25, -0.6)),
    ("Center", (-0.75, 0.0)),
    ("Center left", (-0.75, 1.0)),
    ("Center right", (-0.75, -1.0)),
    ("Center near", (-0.25, 0.0)),
    ("Center approach", (0.25, 0.0)),
    ("Center close", (0.75, 0.0)),
)


@dataclass(frozen=True)
class ExecutionStats:
    frames: int
    elapsed_ms: float
    overrun_steps: int
    collision_detected: bool = False


@dataclass(frozen=True)
class EndEffectorTrackingDiagnostic:
    name: str
    target_w: np.ndarray
    reference_w: np.ndarray


@dataclass(frozen=True)
class DemoRun:
    title: str
    start_xy: tuple[float, float]


class MotionExecutionTimeout(RuntimeError):
    pass


class EpisodeStepLimitReached(RuntimeError):
    pass


class RobotFell(RuntimeError):
    pass


def portrait_corridor_demo_runs(count: int) -> tuple[DemoRun, ...]:
    if not 1 <= count <= len(PORTRAIT_CORRIDOR_RUNS):
        raise ValueError(
            f"Demo run count must be in 1..{len(PORTRAIT_CORRIDOR_RUNS)}, got {count}"
        )
    return tuple(
        DemoRun(title, start_xy) for title, start_xy in PORTRAIT_CORRIDOR_RUNS[:count]
    )


def repeated_demo_runs(count: int) -> tuple[DemoRun, ...]:
    """Return identical reset positions for independently recorded episodes."""
    return tuple(DemoRun(f"Run {index + 1}", (0.0, 0.0)) for index in range(count))


class SimRuntime:
    def __init__(
        self,
        node: Node,
        simulation: MjlabEnv,
        policy: SonicPolicy,
        renderer: SimRenderer,
        viewer: SimViewer | None = None,
        step_hook: TaskStepHook | None = None,
        recorder: DemoVideoRecorder | None = None,
        stop_recording_at_corridor: bool = False,
        stop_recording_at_stand: bool = False,
        motion_timeout_seconds: float = 20.0,
        episode_max_steps: int | None = None,
        demo_runs: tuple[DemoRun, ...] = (),
    ) -> None:
        self.node = node
        self.simulation = simulation
        self.policy = policy
        self.renderer = renderer
        self.viewer = viewer
        self.step_hook = step_hook
        self.recorder = recorder
        self.stop_recording_at_corridor = stop_recording_at_corridor
        self.stop_recording_at_stand = stop_recording_at_stand
        if motion_timeout_seconds <= 0.0:
            raise ValueError("Motion timeout must be positive")
        self.max_motion_frames = max(1, math.ceil(motion_timeout_seconds * SONIC_FPS))
        self.motion_timeout_seconds = motion_timeout_seconds
        if episode_max_steps is not None and episode_max_steps <= 0:
            raise ValueError("Episode step limit must be positive")
        self.episode_max_steps = episode_max_steps
        self.episode_steps = 0
        self.demo_runs = demo_runs or (DemoRun("Center start", (0.0, 0.0)),)
        self.reset_for_demo_run = bool(demo_runs)
        self.run_index = 0
        self.run_id = 0
        self.observation_id = 0
        self.episode_steps = 0
        self._projection_cache: ProjectionContext | None = None
        self._observation_published_at: float | None = None
        self.demo_vlm_state = DemoVlmState()
        self._demo_complete = False
        self._last_command_signature: object | None = None
        self._repeated_command_count = 0

    def run(self) -> None:
        self._start_run()
        for event in self.node:
            if event["type"] == "STOP":
                return
            if event["type"] != "INPUT":
                continue
            if event["id"] == "motion":
                self._accept_motion(event)
            elif event["id"] == "grounding_request":
                self._accept_grounding_request(event)
            if self._demo_complete:
                return

    def _start_run(self) -> None:
        run = self.demo_runs[self.run_index]
        if self.reset_for_demo_run:
            self.simulation.reset_at(*run.start_xy)
            self.policy.reset()
        self.observation_id = 0
        self._observation_published_at = None
        self.demo_vlm_state = DemoVlmState()
        self._last_command_signature = None
        self._repeated_command_count = 0
        render_ms, jpeg_size = self._publish_observation(completed_command=None)
        self.node.log(
            "info",
            f"[RUN {self.run_index + 1}/{len(self.demo_runs)} OBS 0] "
            f"initial observation: render_ms={render_ms:.1f} "
            f"jpeg_kb={jpeg_size / 1024.0:.1f} waiting=motion",
            target="dsrf.sim",
            fields={
                "event": "initial_observation",
                "run": str(self.run_index + 1),
                "observation_id": "0",
                "render_ms": f"{render_ms:.1f}",
                "jpeg_kb": f"{jpeg_size / 1024.0:.1f}",
            },
        )

    def _accept_grounding_request(self, event: dict[str, Any]) -> None:
        try:
            request = grounding_request_from_arrow(
                event["value"], dict(event.get("metadata") or {})
            )
            if request.observation_id != self.observation_id:
                raise ValueError(
                    f"Expected grounding request for observation {self.observation_id}, got {request.observation_id}"
                )
            if self._projection_cache is None:
                self._projection_cache = self.renderer.capture_depth()
            resolved = tuple(
                resolve_waypoint(point, self._projection_cache)
                for point in request.waypoints_2d
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
        result = GroundingResult(
            self.observation_id,
            tuple(item.target_xy for item in resolved),
            tuple(
                EndEffectorTarget(target.name, target.target_xyz)
                for target in resolved_end_effectors
            ),
        )
        data, metadata = grounding_result_to_arrow(result)
        self.node.send_output("grounding_result", data, metadata=metadata)

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
                self.policy.load_motion(chunk, state.root_pos_w, state.root_quat_w)
            except ValueError as exc:
                self._report_error(str(exc))
                return
        diagnostics = self._reference_end_effector_diagnostics(chunk, state)

        self.demo_vlm_state = DemoVlmState(
            observation_id=chunk.observation_id,
            reasoning=chunk.reasoning or "",
            command=chunk.command,
        )
        signature = _command_signature(chunk.command)
        if signature == self._last_command_signature:
            self._repeated_command_count += 1
        else:
            self._last_command_signature = signature
            self._repeated_command_count = 1
        if self.viewer is not None and hasattr(self.viewer, "set_vlm_result"):
            self.viewer.set_vlm_result(
                chunk.observation_id,
                chunk.reasoning,
                chunk.command,
            )

        published_at = self._observation_published_at
        pause_ms = (
            (received_at - published_at) * 1000.0 if published_at is not None else 0.0
        )
        try:
            stats = self._execute()
        except RobotFell as exc:
            self._reset_after_fall(str(exc))
            return
        except MotionExecutionTimeout as exc:
            self._handle_motion_timeout(chunk.command, str(exc))
            return
        except EpisodeStepLimitReached as exc:
            self._handle_episode_step_limit(str(exc))
            return
        if (
            self.episode_max_steps is not None
            and self.episode_steps >= self.episode_max_steps
        ):
            self._handle_episode_step_limit(
                f"Episode reached {self.episode_max_steps} simulation steps"
            )
            return
        if self._repeated_command_count >= MAX_REPEATED_COMMANDS:
            self._handle_repeated_command(chunk.command)
            return
        if self._stop_recording_if_ready(chunk.command):
            return
        if self.stop_recording_at_stand and _is_stand_command(chunk.command):
            self._advance_or_stop("standing motion completed")
            return
        self._log_end_effector_tracking(diagnostics)
        completed_observation_id = self.observation_id
        self.observation_id += 1
        render_ms, jpeg_size = self._publish_observation(
            completed_command=chunk.command,
            collision_detected=stats.collision_detected,
        )
        target_ms = stats.frames * CONTROL_PERIOD * 1000.0
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
        collision_detected = False
        with torch.no_grad():
            while True:
                if (
                    self.episode_max_steps is not None
                    and self.episode_steps >= self.episode_max_steps
                ):
                    raise EpisodeStepLimitReached(
                        f"Episode reached {self.episode_max_steps} simulation steps"
                    )
                if frames >= self.max_motion_frames:
                    raise MotionExecutionTimeout(
                        f"Motion exceeded {self.motion_timeout_seconds:.1f}s "
                        f"({self.max_motion_frames} simulation frames)"
                    )
                delay = next_step - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)

                with self.simulation.compute_context():
                    state = self.simulation.robot_state()
                    action, completed = self.policy.infer(state)
                if self.step_hook is not None:
                    self.step_hook.before_step()
                self.simulation.step(action)
                fall_detector = getattr(self.simulation, "fall_reason", None)
                if fall_detector is not None:
                    fall_reason = fall_detector()
                    if fall_reason is not None:
                        raise RobotFell(fall_reason)
                collision_detector = getattr(
                    self.simulation, "task_collision_detected", None
                )
                collision_detected |= (
                    bool(collision_detector())
                    if collision_detector is not None
                    else False
                )
                if self.viewer is not None:
                    self.viewer.sync()
                if self.recorder is not None:
                    self.recorder.write_frame(
                        self.renderer.capture_demo_rgb(), self.demo_vlm_state
                    )
                frames += 1
                self.episode_steps += 1

                if completed:
                    return ExecutionStats(
                        frames=frames,
                        elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
                        overrun_steps=overrun_steps,
                        collision_detected=collision_detected,
                    )

                next_step += CONTROL_PERIOD
                now = time.perf_counter()
                if next_step < now:
                    overrun_steps += 1
                    next_step = now

    def _reference_end_effector_diagnostics(
        self, chunk: MotionChunk, state: Any
    ) -> tuple[EndEffectorTrackingDiagnostic, ...]:
        if not chunk.end_effectors:
            return ()
        root_pos_w = state.root_pos_w
        root_quat_w = state.root_quat_w
        names = tuple(target.name for target in chunk.end_effectors)
        targets_w = self.simulation.end_effector_targets_w(
            chunk.end_effectors,
            start_root_pos_w=root_pos_w,
            start_root_quat_w=root_quat_w,
        )
        reference_w = self.simulation.reference_end_effector_positions_w(
            chunk.qpos[-1],
            start_root_pos_w=root_pos_w,
            start_root_quat_w=root_quat_w,
            names=names,
        )
        return tuple(
            EndEffectorTrackingDiagnostic(name, targets_w[name], reference_w[name])
            for name in names
        )

    def _log_end_effector_tracking(
        self, diagnostics: tuple[EndEffectorTrackingDiagnostic, ...]
    ) -> None:
        if not diagnostics:
            return
        actual_w = self.simulation.end_effector_positions_w(
            tuple(item.name for item in diagnostics)
        )
        for item in diagnostics:
            reference_error_cm = float(
                np.linalg.norm(item.reference_w - item.target_w) * 100.0
            )
            actual_error_cm = float(
                np.linalg.norm(actual_w[item.name] - item.target_w) * 100.0
            )
            self.node.log(
                "info",
                f"[OBS {self.observation_id}] {item.name} target error: "
                f"ARDY reference={reference_error_cm:.1f} cm, "
                f"SONIC actual={actual_error_cm:.1f} cm",
                target="dsrf.sim.end_effector",
                fields={
                    "event": "end_effector_tracking",
                    "observation_id": str(self.observation_id),
                    "end_effector": item.name,
                    "reference_error_cm": f"{reference_error_cm:.3f}",
                    "actual_error_cm": f"{actual_error_cm:.3f}",
                },
            )

    def _stop_recording_if_ready(self, command: str) -> bool:
        if (
            self.recorder is None
            or not self.stop_recording_at_corridor
            or not _is_stand_command(command)
        ):
            return False
        with self.simulation.compute_context():
            root_x = float(self.simulation.robot_state().root_pos_w[0].item())
        if root_x < PORTRAIT_CORRIDOR_APPROACH_X:
            return False

        self.node.log(
            "info",
            "Demo run completed: robot is standing at the corridor approach "
            f"(x={root_x:.2f})",
            target="dsrf.sim",
            fields={
                "event": "demo_run_completed",
                "run": str(self.run_index + 1),
                "root_x": f"{root_x:.3f}",
                "command": command,
            },
        )
        self._advance_or_stop("corridor reached")
        return True

    def _handle_motion_timeout(self, command: str, detail: str) -> None:
        self.node.log(
            "error",
            f"[OBS {self.observation_id}] motion timeout: {detail}",
            target="dsrf.sim",
            fields={
                "event": "motion_timeout",
                "observation_id": str(self.observation_id),
                "command": command,
                "detail": detail,
            },
        )
        self._advance_or_stop("motion timeout")

    def _reset_after_fall(self, reason: str) -> None:
        """Start a clean episode and surface the physical failure to the VLM."""
        self.node.log(
            "warning",
            f"[OBS {self.observation_id}] robot fell; resetting episode: {reason}",
            target="dsrf.sim",
            fields={
                "event": "episode_reset_after_fall",
                "observation_id": str(self.observation_id),
                "reason": reason,
            },
        )
        self.simulation.reset()
        self.policy.reset()
        if self.step_hook is not None and hasattr(self.step_hook, "reset"):
            self.step_hook.reset()
        self.run_id += 1
        self.observation_id = 0
        self.episode_steps = 0
        self._projection_cache = None
        self._observation_published_at = None
        self.demo_vlm_state = DemoVlmState()
        self._last_command_signature = None
        self._repeated_command_count = 0
        self._publish_observation(completed_command=None, reset_reason=reason)

    def _handle_episode_step_limit(self, detail: str) -> None:
        self.node.log(
            "info",
            f"[RUN {self.run_index + 1}] episode step limit reached: {detail}",
            target="dsrf.sim",
            fields={
                "event": "episode_step_limit",
                "run": str(self.run_index + 1),
                "detail": detail,
            },
        )
        self._advance_or_stop("episode step limit")

    def _handle_repeated_command(self, command: str) -> None:
        self.node.log(
            "error",
            f"[OBS {self.observation_id}] repeated command loop: "
            f"{command!r} repeated {self._repeated_command_count} times",
            target="dsrf.sim",
            fields={
                "event": "repeated_command_loop",
                "observation_id": str(self.observation_id),
                "command": command,
                "count": str(self._repeated_command_count),
            },
        )
        self._advance_or_stop("repeated command loop")

    def _advance_or_stop(self, reason: str) -> None:
        if self.run_index + 1 < len(self.demo_runs):
            self.run_index += 1
            self.run_id += 1
            self.node.log(
                "info",
                f"Starting demo run {self.run_index + 1}/{len(self.demo_runs)} "
                f"after {reason}",
                target="dsrf.sim",
                fields={
                    "event": "demo_next_run",
                    "run": str(self.run_index + 1),
                    "reason": reason,
                },
            )
            self._start_run()
            return
        self._close_recorder()
        self._demo_complete = True
        self._request_dataflow_stop()

    def _close_recorder(self) -> None:
        if self.recorder is None:
            return
        recorder = self.recorder
        self.recorder = None
        recorder.close()
        self.node.log(
            "info",
            "Demo recording stopped",
            target="dsrf.sim",
            fields={"event": "demo_recording_stopped"},
        )

    def _request_dataflow_stop(self) -> None:
        command = ["dora", "stop"]
        dataflow_id = _current_dataflow_id()
        if dataflow_id is not None:
            command.append(dataflow_id)
        command.extend(("--grace-duration", "5s"))
        try:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.node.log(
                "info",
                "Dataflow stop requested",
                target="dsrf.sim",
                fields={
                    "event": "demo_dataflow_stop_requested",
                    "dataflow_id": dataflow_id or "auto",
                },
            )
        except OSError as exc:
            self.node.log(
                "error",
                f"Failed to request dataflow stop: {exc}",
                target="dsrf.sim",
                fields={
                    "event": "demo_dataflow_stop_error",
                    "detail": str(exc),
                },
            )

    def _publish_observation(
        self,
        *,
        completed_command: str | None,
        collision_detected: bool = False,
        reset_reason: str | None = None,
    ) -> tuple[float, int]:
        render_started_at = time.perf_counter()
        jpeg = self.renderer.capture_jpeg()
        render_ms = (time.perf_counter() - render_started_at) * 1000.0
        observation = VisualObservation(
            observation_id=self.observation_id,
            completed_command=completed_command,
            jpeg=jpeg,
            run_id=self.run_id,
            collision_detected=collision_detected,
            reset_reason=reset_reason,
        )
        data, metadata = observation_to_arrow(observation)
        self.node.send_output("observation", data, metadata=metadata)
        self._observation_published_at = time.perf_counter()
        if self.viewer is not None and hasattr(self.viewer, "set_vlm_thinking"):
            self.viewer.set_vlm_thinking(self.observation_id)
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


def _is_stand_command(command: str) -> bool:
    try:
        payload = json.loads(command)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("motion") == "stand"


def _command_signature(command: str) -> object:
    try:
        payload = json.loads(command)
    except json.JSONDecodeError:
        return command.strip()
    if not isinstance(payload, dict):
        return command.strip()
    motion = payload.get("motion")
    if motion in {"walk", "turn"} and "direction" in payload:
        return (motion, payload.get("direction"))
    if motion == "walk" and "waypoints_2d" in payload:
        waypoints = payload.get("waypoints_2d")
        if isinstance(waypoints, list):
            return ("walk", tuple(tuple(waypoint) for waypoint in waypoints))
    return (motion,)


def _current_dataflow_id() -> str | None:
    """Read the current UUID from the config Dora injects into each node."""
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
