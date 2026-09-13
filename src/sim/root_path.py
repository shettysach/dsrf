"""Measured-state, timed root path for the scripted box push."""

import math

import numpy as np
from tasks.push_motion.settings import PushMotionSettings

from motion_gen.targets import TimedTargets
from shared.messages import EndEffectorTarget, RootPathGoal


class RootPathController:
    """Keep timed root and sparse hand goals tied to measured progress."""

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
        self.frame = 0
        self.deadline: int | None = None
        self.fps = 25.0

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
        self.deadline = None

    def fail(self, reason: str) -> None:
        self.phase, self.reason = "failed", reason

    def update(
        self,
        qpos: np.ndarray,
        dt: float,
        hands: dict[str, np.ndarray] | None = None,
        box_position: np.ndarray | None = None,
    ) -> bool:
        if self.finished:
            return True
        self.frame += round(dt * self.fps)
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
            if qpos[2] < 0.55:
                self.fail("Root fell below 0.55 m at the goal")
            elif (
                self._hands_forward(qpos, hands)
                and box_position is not None
                and box_position[0] >= self.config.box_goal_x - 0.10
                and abs(box_position[1]) <= 0.15
            ):
                self.phase, self.reason = "done", "Box reached the goal with palms forward"

        return self.finished

    @staticmethod
    def _hands_forward(qpos: np.ndarray, hands: dict[str, np.ndarray] | None) -> bool:
        if hands is None or set(hands) != {"left_hand", "right_hand"}:
            return False
        w, x, y, z = qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        forward = np.array((np.cos(yaw), np.sin(yaw)))
        return all(
            float(np.dot(hand[:2] - qpos[:2], forward)) >= 0.20
            for hand in hands.values()
        )

    def targets(
        self, qpos: np.ndarray, frames: int, fps: float, visible_frames: int = 52
    ) -> tuple[TimedTargets, ...]:
        self.fps = fps
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
        root_frames = {frames - 1}
        if self.phase != "reach":
            if self.deadline is None or self.deadline < self.frame:
                self.deadline = (
                    self.frame
                    + max(frames, math.ceil(self.remaining(qpos) * fps / speed))
                    - 1
                )
            goal_frame = self.deadline - self.frame
            if 0 <= goal_frame < visible_frames:
                root_frames.add(goal_frame)
        hand_frames = (
            {frames - 1}
            if self.phase in {"reach", "push"} and self.config.hand_targets
            else set()
        )
        samples = []
        for frame in sorted(root_frames | hand_frames):
            root_delta = np.zeros(3)
            if self.phase != "reach":
                target = (
                    self.goal.approach_xy
                    if self.phase == "approach"
                    else self.goal.target_xy
                )
                delta = np.asarray(target) - root[:2]
                distance = float(np.linalg.norm(delta))
                fraction = (
                    1.0
                    if frame != frames - 1
                    or (
                        self.deadline is not None
                        and self.deadline - self.frame <= frames - 1
                    )
                    else min(1.0, speed * ((frame + 1) / fps) / max(distance, 1e-8))
                )
                root_delta[:2] = delta * fraction
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
                    (float(local_delta[0]), float(local_delta[1]))
                    if frame in root_frames
                    else None,
                    hands,
                )
            )
        return tuple(samples)
