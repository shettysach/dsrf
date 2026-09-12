from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml
from dora import Node
from tasks.box_push.settings import BoxPushSettings

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
from sim.push import PushController
from sim.renderer import SimRenderer
from sim.root_path import RootPathController
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
        publish_observations: bool = True,
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
        self.publish_observations = publish_observations
        self.completed_commands = 0
        self.demo_vlm_state = DemoVlmState()
        self._demo_observation_rgb: np.ndarray | None = None
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
            if self.publish_observations:
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
        # The VLM selected its targets on this exact observation. Record it now
        # that the response is available, before generating or stepping motion.
        if self.recorder is not None and self._demo_observation_rgb is not None:
            self.recorder.write_frame(self._demo_observation_rgb, self.demo_vlm_state)

        if command.contact_goal is not None:
            self._execute_push(command)
            return
        if command.root_path_goal is not None:
            self._execute_root_path(command)
            return

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

        # A failed generation leaves this observation current, so keep its
        # synchronized depth available for a corrected command. Once generation
        # succeeds, the next physics step can change the scene and invalidates it.
        self._projection_cache = None
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
        sokoban_events = self.simulation.begin_sokoban_motion_events()
        stats = self._execute(
            after_step=(
                (lambda: self.simulation.observe_sokoban_motion_events(sokoban_events))
                if sokoban_events is not None
                else None
            )
        )
        execution_feedback = (
            self.simulation.finish_sokoban_motion_events(sokoban_events)
            if sokoban_events is not None
            else None
        )
        completed_observation_id = self.observation_id
        self.completed_commands += 1
        if getattr(self.simulation, "task_completed", False):
            self._stop_requested = True
            self.node.log(
                "info",
                f"[OBS {completed_observation_id}] task completion detected",
                target="dsrf.sim",
                fields={
                    "event": "task_completed",
                    "observation_id": str(completed_observation_id),
                },
            )
            return
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
        if not self.publish_observations:
            return
        render_ms, jpeg_size = self._publish_observation(
            completed_command=command.text,
            execution_feedback=execution_feedback,
        )
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

    def _execute_push(self, command: AgentCommand) -> None:
        """Execute a scripted contact goal without publishing intermediate observations."""
        from motion_gen.ardy.adapter import ArdyMotionGenerator

        goal = command.contact_goal
        assert goal is not None
        settings = BoxPushSettings.from_env()
        if not isinstance(self.generator, ArdyMotionGenerator):
            self._report_error("Scripted contact requires ARDY", source="execution")
            self._stop_requested = True

            return
        generator = self.generator
        windows = 0
        try:
            with self.simulation.compute_context():
                state = self.simulation.push_state(goal.body)
            controller = PushController(
                goal,
                state,
                window_seconds=generator.window_frames / generator.fps,
                config=settings,
            )
            history = deque([state.qpos.copy() for _ in range(9)], maxlen=9)
            if not np.isclose(self.simulation.step_dt, 0.02) or generator.fps != 25:
                raise ValueError(
                    "Scripted history sampling requires 50 Hz sim / 25 Hz ARDY"
                )

            def after_step() -> bool:
                nonlocal state
                with self.simulation.compute_context():
                    state = self.simulation.push_state(goal.body)
                history.append(state.qpos.copy())
                previous = controller.phase
                interrupt = controller.update(state, self.simulation.step_dt)
                if controller.phase == "push" and previous != "push":
                    self.node.log(
                        "info",
                        "Pushing after sustained physical contact",
                        target="dsrf.sim.push",
                    )
                if controller.phase != previous:
                    self.node.log(
                        "info",
                        f"Push phase: {previous} -> {controller.phase}",
                        target="dsrf.sim.push",
                    )
                return interrupt or self._stop_requested

            while not controller.finished and not self._stop_requested:
                motion = controller.motion_prompt
                self.demo_vlm_state = DemoVlmState(
                    observation_id=command.observation_id,
                    reasoning=f"Script: {controller.phase}; window {windows + 1}; "
                    f"remaining {controller.remaining(state):.2f}m; contacts {sorted(state.contacts)}",
                    command=motion,
                )
                samples = controller.targets(
                    state, generator.window_frames, generator.fps
                )
                started = time.perf_counter()
                qpos, reference = self._generate_scripted_window(
                    generator, motion, samples, history, load_virtual_force=True
                )
                windows += 1
                self.node.log(
                    "info",
                    f"Push window {windows}: phase={controller.phase} "
                    f"remaining={controller.remaining(state):.3f}m "
                    f"contacts={sorted(state.contacts)}",
                    target="dsrf.sim.push",
                    fields={
                        "event": "push_window",
                        "window": str(windows),
                        "phase": controller.phase,
                        "remaining_m": str(controller.remaining(state)),
                    },
                )
                self._log_motion_generated(
                    command,
                    qpos,
                    reference,
                    plan_ms=(time.perf_counter() - started) * 1000,
                    prompt=motion,
                )
                self._execute(after_step=after_step)

            success = controller.phase == "done"
            self.node.log(
                "info" if success else "error",
                f"Scripted push {'succeeded' if success else 'failed'}: "
                f"{controller.reason or 'Stopped externally'}; windows={windows}, "
                f"sim_seconds={controller.elapsed:.2f}, remaining={controller.remaining(state):.3f}m",
                target="dsrf.sim.push",
                fields={
                    "event": "push_result",
                    "success": str(success).lower(),
                    "reason": controller.reason,
                    "windows": str(windows),
                    "sim_seconds": str(controller.elapsed),
                },
            )
            if success:
                self.completed_commands += 1
            else:
                self._report_error(
                    controller.reason or "Stopped externally", source="execution"
                )
        except (ValueError, KeyError) as exc:
            self._report_error(str(exc), source="execution")
        finally:
            # This is the single-interaction script path, not a VLM protocol.
            self._stop_requested = True

    def _execute_root_path(self, command: AgentCommand) -> None:
        """Run the old multi-window script shape with root targets only."""
        from motion_gen.ardy.adapter import ArdyMotionGenerator

        goal = command.root_path_goal
        assert goal is not None
        if not isinstance(self.generator, ArdyMotionGenerator):
            self._report_error("Scripted root path requires ARDY", source="execution")
            self._stop_requested = True
            return
        generator = self.generator
        windows = 0
        try:
            with self.simulation.compute_context():
                state = self._root_path_qpos()
            controller = RootPathController(
                goal, window_seconds=generator.window_frames / generator.fps
            )
            history = deque([state.copy() for _ in range(9)], maxlen=9)
            if not np.isclose(self.simulation.step_dt, 0.02) or generator.fps != 25:
                raise ValueError(
                    "Scripted history sampling requires 50 Hz sim / 25 Hz ARDY"
                )

            def after_step() -> bool:
                nonlocal state
                with self.simulation.compute_context():
                    state = self._root_path_qpos()
                history.append(state.copy())
                previous = controller.phase
                interrupt = controller.update(state, self.simulation.step_dt)
                if controller.phase != previous:
                    self.node.log(
                        "info",
                        f"Root-path phase: {previous} -> {controller.phase}",
                        target="dsrf.sim.root_path",
                    )
                return interrupt or self._stop_requested

            while not controller.finished and not self._stop_requested:
                motion = controller.motion_prompt
                samples = controller.targets(
                    state, generator.window_frames, generator.fps
                )
                self.demo_vlm_state = DemoVlmState(
                    observation_id=command.observation_id,
                    reasoning=(
                        f"Script: {controller.phase}; window {windows + 1}; "
                        f"remaining {controller.remaining(state):.2f}m"
                    ),
                    command=motion,
                )
                started = time.perf_counter()
                qpos, reference = self._generate_scripted_window(
                    generator, motion, samples, history, load_virtual_force=False
                )
                windows += 1
                self._log_motion_generated(
                    command,
                    qpos,
                    reference,
                    plan_ms=(time.perf_counter() - started) * 1000,
                    prompt=motion,
                )
                self._execute(after_step=after_step)

            success = controller.phase == "done"
            self.node.log(
                "info" if success else "error",
                f"Scripted root path {'succeeded' if success else 'failed'}: "
                f"{controller.reason or 'Stopped externally'}; windows={windows}",
                target="dsrf.sim.root_path",
            )
            if success:
                self.completed_commands += 1
            else:
                self._report_error(
                    controller.reason or "Stopped externally", source="execution"
                )
        except ValueError as exc:
            self._report_error(str(exc), source="execution")
        finally:
            self._stop_requested = True

    def _root_path_qpos(self) -> np.ndarray:
        state = self.simulation.robot_state()
        qpos = torch.cat((state.root_pos_w, state.root_quat_w, state.joint_pos))
        return qpos.detach().cpu().numpy().copy()

    def _generate_scripted_window(
        self,
        generator: Any,
        motion: str,
        samples: Any,
        history: deque[np.ndarray],
        *,
        load_virtual_force: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep an extra observed frame for velocity encoding, then crop the
        # encoded sequence to four frames inside ARDY.observe().
        observed = np.stack(list(history)[::2])
        with self.simulation.compute_context():
            source_qpos = generator.generate_window(motion, samples, observed)
            reference = resample_qpos(source_qpos, source_fps=generator.fps)
            state = self.simulation.robot_state()
            self.tracker.load_motion(reference, state, world_aligned=True)
            if load_virtual_force and self.virtual_force is not None:
                self.virtual_force.load_motion(reference, state)
        self._projection_cache = None
        return source_qpos, reference

    def _execute(
        self, *, after_step: Callable[[], bool] | None = None
    ) -> ExecutionStats:
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
                frames += 1
                if self.recorder is not None and getattr(
                    self.recorder, "should_capture", lambda _: True
                )(frames):
                    self.recorder.write_frame(
                        self.renderer.capture_demo_rgb(), self.demo_vlm_state
                    )

                interrupted = after_step() if after_step is not None else False

                # Completion is detected while producing the last reference
                # action. Capture only after that action's physics step.
                if completed or interrupted or self._stop_requested:
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
        self,
        *,
        completed_command: str | None,
        execution_feedback: str | None = None,
    ) -> tuple[float, int]:
        render_started_at = time.perf_counter()
        if self.recorder is None:
            jpeg, projection = self.renderer.capture_rgbd()
            self._demo_observation_rgb = None
        else:
            jpeg, projection, self._demo_observation_rgb = (
                self.renderer.capture_observation()
            )
        render_ms = (time.perf_counter() - render_started_at) * 1000.0
        observation = VisualObservation(
            observation_id=self.observation_id,
            completed_command=completed_command,
            jpeg=jpeg,
            execution_feedback=execution_feedback,
        )
        self._projection_cache = projection
        self._observation_published_at = time.perf_counter()
        data, metadata = observation_to_arrow(observation)
        self.node.send_output("observation", data, metadata=metadata)
        return render_ms, len(jpeg)

    def _log_motion_generated(
        self,
        command: AgentCommand,
        source_qpos: torch.Tensor,
        qpos: torch.Tensor,
        *,
        plan_ms: float,
        prompt: str | None = None,
    ) -> None:
        duration_s = len(qpos) / REFERENCE_HZ
        motion_prompt = prompt or command.text
        self.node.log(
            "info",
            f"[OBS {command.observation_id}] motion generated: "
            f"command={motion_prompt!r} frames={len(qpos)} "
            f"duration_s={duration_s:.2f} plan_ms={plan_ms:.1f}",
            target="dsrf.motion_gen",
            fields={
                "event": "motion_generated",
                "observation_id": str(command.observation_id),
                "command": motion_prompt,
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
