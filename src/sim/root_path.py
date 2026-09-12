"""Feedback-paced root-only path controller for scripted ARDY motions."""

import numpy as np
from tasks.push_motion.settings import PushMotionSettings

from motion_gen.targets import TimedTargets
from shared.messages import EndEffectorTarget, RootPathGoal

_REACH_HAND_FRAMES = frozenset({15, 31, 51})
_PUSH_HAND_FRAMES = frozenset({9, 19, 29, 39, 51})

class RootPathController:
    """Own root progress while deliberately supplying no body-pose targets."""

    def __init__(
        self,
        goal: RootPathGoal,
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

    def remaining(self, qpos: np.ndarray) -> float:
        target = (
            self.goal.approach_xy if self.phase == "approach" else self.goal.target_xy
        )
        return float(np.linalg.norm(np.asarray(target) - qpos[:2]))

    def _transition(self, phase: str) -> None:
        self.phase = phase
        self.phase_elapsed = 0.0

    def fail(self, reason: str) -> None:
        self.phase, self.reason = "failed", reason

    def reference_failure(self, qpos: np.ndarray) -> str | None:
        if not np.isfinite(qpos).all():
            return "Non-finite generated reference"
        if float(np.min(qpos[:, 2])) < self.config.min_root_height:
            return "Generated reference drops below the root-height limit"
        if _max_tilt_degrees(qpos[:, 3:7]) > self.config.max_reference_tilt_degrees:
            return "Generated reference exceeds the tilt limit"
        return None

    def tracking_failure(self, actual_qpos: np.ndarray, _reference_qpos: np.ndarray) -> str | None:
        if float(actual_qpos[2]) < self.config.min_root_height:
            return "Robot drops below the root-height limit"
        if _max_tilt_degrees(actual_qpos[None, 3:7]) > self.config.max_tilt_degrees:
            return "Robot exceeds the tilt limit"
        return None

    def update(self, qpos: np.ndarray, dt: float) -> bool:
        if self.finished:
            return True
        self.elapsed += dt
        self.phase_elapsed += dt
        if not np.isfinite(qpos).all():
            self.phase, self.reason = "failed", "Non-finite robot state"
        elif self.elapsed >= self.config.timeout:
            self.phase, self.reason = "failed", "Motion time budget exhausted"
        elif self.phase == "approach" and self.remaining(qpos) <= 0.10:
            self._transition("reach")
        elif self.phase == "reach" and (
            self.phase_elapsed >= self.config.reach_windows * self.window_seconds
        ):
            self._transition("push")
        elif self.phase == "push" and self.remaining(qpos) <= 0.10:
            self.phase, self.reason = "done", "Reached the root-path goal"

        return self.finished

    def targets(
        self, qpos: np.ndarray, frames: int, fps: float
    ) -> tuple[TimedTargets, ...]:
        root = qpos[:3]
        w, x, y, z = qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        speed = (
            self.config.navigation_speed
            if self.phase == "approach"
            else self.config.push_speed
        )
        root_frames = set(range(3, frames, 4)) | {frames - 1}
        hand_frames = (
            _REACH_HAND_FRAMES
            if self.phase == "reach" and self.config.hand_targets
            else _PUSH_HAND_FRAMES
            if self.phase == "push" and self.config.hand_targets
            else frozenset()
        )
        samples = []
        for frame in sorted(
            root_frames | {frame for frame in hand_frames if frame < frames}
        ):
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
            hands = (
                tuple(
                    EndEffectorTarget(
                        point.name,
                        tuple(np.asarray(point.target_xyz) + local_delta),
                        palm_normal=point.palm_normal,
                    )
                    for point in self.goal.points
                )
                if frame in hand_frames
                else ()
            )
            samples.append(
                TimedTargets(
                    frame,
                    (float(local_delta[0]), float(local_delta[1])),
                    hands,
                )
            )
        return tuple(samples)


def _max_tilt_degrees(quaternions: np.ndarray) -> float:
    """Return the largest angle between the root's up axis and world up."""
    x, y = quaternions[:, 1], quaternions[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))).max())
