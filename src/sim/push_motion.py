"""Feedback controller for a contact-free bilateral push motion."""

from dataclasses import dataclass

import numpy as np
from tasks.push_motion.settings import PushMotionSettings

from motion_gen.targets import TimedTargets
from shared.messages import EndEffectorTarget, PushMotionGoal


@dataclass(frozen=True)
class PushMotionState:
    qpos: np.ndarray


class PushMotionController:
    """Advance a virtual push pose without inspecting scene contact or objects."""

    def __init__(
        self,
        goal: PushMotionGoal,
        state: PushMotionState,
        *,
        window_seconds: float,
        config: PushMotionSettings | None = None,
    ) -> None:
        self.goal = goal
        self.config = config or PushMotionSettings.from_env()
        self.window_seconds = window_seconds
        self.phase = "approach"
        self.elapsed = self.phase_elapsed = 0.0
        self.reason = ""
        self._progress_time = 0.0
        self._progress_distance = self._distance_for_phase(state)

    @property
    def finished(self) -> bool:
        return self.phase in {"done", "failed"}

    @property
    def motion_prompt(self) -> str:
        return {
            "approach": self.config.approach_prompt,
            "reach": self.config.reach_prompt,
            "push": self.config.push_prompt,
        }[self.phase]

    def _root_xy(self, state: PushMotionState) -> np.ndarray:
        return state.qpos[:2]

    def _distance_for_phase(self, state: PushMotionState) -> float:
        destination = (
            self.goal.approach_xy if self.phase == "approach" else self.goal.target_xy
        )
        return float(np.linalg.norm(np.asarray(destination) - self._root_xy(state)))

    def remaining(self, state: PushMotionState) -> float:
        """Distance to the current virtual phase destination."""
        return self._distance_for_phase(state)

    def _transition(self, phase: str, state: PushMotionState) -> None:
        self.phase = phase
        self.phase_elapsed = 0.0
        self._progress_time = self.elapsed
        self._progress_distance = self._distance_for_phase(state)

    def fail(self, reason: str) -> None:
        self.phase, self.reason = "failed", reason

    def update(self, state: PushMotionState, dt: float) -> bool:
        """Track virtual spatial progress; no contacts are read or required."""
        if self.finished:
            return True
        self.elapsed += dt
        self.phase_elapsed += dt
        if not np.isfinite(state.qpos).all():
            self.fail("Non-finite robot state")
        elif self.elapsed >= self.config.timeout:
            self.fail("Motion time budget exhausted")
        elif self.phase == "approach" and self._distance_for_phase(state) <= 0.10:
            self._transition("reach", state)
        elif self.phase == "reach" and (
            self.phase_elapsed >= self.config.reach_windows * self.window_seconds
        ):
            self._transition("push", state)
        elif self.phase == "push" and self._distance_for_phase(state) <= 0.10:
            self.phase, self.reason = "done", "Reached the virtual push goal"

        if self.phase in {"approach", "push"}:
            distance = self._distance_for_phase(state)
            if self._progress_distance - distance >= self.config.progress_distance:
                self._progress_time, self._progress_distance = self.elapsed, distance
            elif self.elapsed - self._progress_time >= self.config.stall_seconds:
                self.fail("No meaningful base progress")
        return self.finished

    def targets(
        self, state: PushMotionState, frames: int, fps: float
    ) -> tuple[TimedTargets, ...]:
        root = state.qpos[:3]
        w, x, y, z = state.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        samples: list[TimedTargets] = []
        for frame in sorted(set(range(3, frames, 4)) | {frames - 1}):
            t = (frame + 1) / fps
            root_delta = np.zeros(3)
            hands: tuple[EndEffectorTarget, ...] = ()
            if self.phase == "approach":
                delta = np.asarray(self.goal.approach_xy) - root[:2]
                distance = float(np.linalg.norm(delta))
                root_delta[:2] = delta * min(
                    1.0, self.config.navigation_speed * t / max(distance, 1e-8)
                )
            elif self.phase == "reach":
                hands = self.goal.points
            elif self.phase == "push":
                delta = np.asarray(self.goal.target_xy) - root[:2]
                distance = float(np.linalg.norm(delta))
                root_delta[:2] = delta * min(
                    1.0, self.config.push_speed * t / max(distance, 1e-8)
                )
                local_delta = world_to_local @ root_delta
                hands = tuple(
                    EndEffectorTarget(
                        point.name,
                        tuple(np.asarray(point.target_xyz) + local_delta),
                        palm_normal=point.palm_normal,
                    )
                    for point in self.goal.points
                )
            local_root = world_to_local @ root_delta
            samples.append(
                TimedTargets(
                    frame,
                    (float(local_root[0]), float(local_root[1])),
                    hands,
                    # The reach is stationary, so an upright torso is useful.
                    # While walking, preserve ARDY's learned pelvis pitch/roll
                    # for balance instead of forcing a rigid upright root.
                    root_upright=self.phase == "reach",
                )
            )
        return tuple(samples)
