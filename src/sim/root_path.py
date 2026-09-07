"""Feedback-paced root-only path controller for scripted ARDY motions."""

from dataclasses import dataclass

import numpy as np
from tasks.push_motion.settings import PushMotionSettings

from motion_gen.targets import TimedTargets
from shared.messages import RootPathGoal


@dataclass(frozen=True)
class RootPathState:
    qpos: np.ndarray


class RootPathController:
    """Own root progress while deliberately supplying no body-pose targets."""

    def __init__(
        self,
        goal: RootPathGoal,
        state: RootPathState,
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

    def remaining(self, state: RootPathState) -> float:
        target = (
            self.goal.approach_xy if self.phase == "approach" else self.goal.target_xy
        )
        return float(np.linalg.norm(np.asarray(target) - state.qpos[:2]))

    def _transition(self, phase: str, state: RootPathState) -> None:
        self.phase = phase
        self.phase_elapsed = 0.0

    def update(self, state: RootPathState, dt: float) -> bool:
        if self.finished:
            return True
        self.elapsed += dt
        self.phase_elapsed += dt
        if not np.isfinite(state.qpos).all():
            self.phase, self.reason = "failed", "Non-finite robot state"
        elif self.elapsed >= self.config.timeout:
            self.phase, self.reason = "failed", "Motion time budget exhausted"
        elif self.phase == "approach" and self.remaining(state) <= 0.10:
            self._transition("reach", state)
        elif self.phase == "reach" and (
            self.phase_elapsed >= self.config.reach_windows * self.window_seconds
        ):
            self._transition("push", state)
        elif self.phase == "push" and self.remaining(state) <= 0.10:
            self.phase, self.reason = "done", "Reached the root-path goal"

        return self.finished

    def targets(
        self, state: RootPathState, frames: int, fps: float
    ) -> tuple[TimedTargets, ...]:
        root = state.qpos[:3]
        w, x, y, z = state.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        speed = (
            self.config.navigation_speed
            if self.phase == "approach"
            else self.config.push_speed
        )
        samples = []
        for frame in sorted(set(range(3, frames, 4)) | {frames - 1}):
            root_delta = np.zeros(3)
            if self.phase != "reach":
                target = (
                    self.goal.approach_xy
                    if self.phase == "approach"
                    else self.goal.target_xy
                )
                delta = np.asarray(target) - root[:2]
                distance = float(np.linalg.norm(delta))
                root_delta[:2] = delta * min(
                    1.0,
                    speed * ((frame + 1) / fps) / max(distance, 1e-8),
                )
            local_delta = world_to_local @ root_delta
            samples.append(
                TimedTargets(
                    frame,
                    (float(local_delta[0]), float(local_delta[1])),
                )
            )
        return tuple(samples)
