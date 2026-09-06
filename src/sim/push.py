"""Small feedback controller for the scripted planar push, independent of ARDY."""

from dataclasses import dataclass

import numpy as np
from tasks.box_push.settings import BoxPushSettings

from motion_gen.targets import TimedTargets
from shared.messages import ContactGoal, EndEffectorTarget


@dataclass(frozen=True)
class PushState:
    qpos: np.ndarray
    body_position: np.ndarray
    body_rotation: np.ndarray
    body_velocity: np.ndarray
    hands: dict[str, np.ndarray]
    contacts: frozenset[str]
    footprint: np.ndarray


class PushController:
    """Own task progress; window exhaustion never implies physical success."""

    def __init__(
        self,
        goal: ContactGoal,
        state: PushState,
        *,
        window_seconds: float,
        config: BoxPushSettings | None = None,
    ) -> None:
        self.goal, self.config = goal, config or BoxPushSettings.from_env()
        self.window_seconds = window_seconds
        self.phase = "approach"
        self.elapsed = self.phase_elapsed = 0.0
        self.contact_age = self.loss_age = self.settle_age = 0.0
        self.retries = 0
        self.reason = ""
        self.attached = False
        self._hand_start = state.hands
        self._root_offset = state.qpos[:2] - state.body_position[:2]
        self._progress_time = 0.0
        self._progress_distance = self.remaining(state)

    @property
    def finished(self) -> bool:
        return self.phase in {"done", "failed"}

    @property
    def motion_prompt(self) -> str:
        """A minimal ARDY prior; spatial constraints supply the task detail."""
        return self.config.prompt_for_phase(self.phase)

    def remaining(self, state: PushState) -> float:
        return float(
            np.linalg.norm(np.asarray(self.goal.target_xy) - state.body_position[:2])
        )

    def contact_points(self, state: PushState) -> dict[str, np.ndarray]:
        return {
            p.name: state.body_position + state.body_rotation @ np.asarray(p.target_xyz)
            for p in self.goal.points
        }

    def begin_push(self, state: PushState, *, maintained_contact: bool) -> None:
        if self.phase != "contact":
            raise ValueError("Push may only begin after contact acquisition")
        self.attached = maintained_contact
        self._transition("push", state)

    def _direction(self, state: PushState) -> np.ndarray:
        delta = np.asarray(self.goal.target_xy) - state.body_position[:2]
        return delta / max(float(np.linalg.norm(delta)), 1e-8)

    def staging_point(self, state: PushState) -> np.ndarray:
        center = np.mean(list(self.contact_points(state).values()), axis=0)
        return center[:2] - self.config.standoff * self._direction(state)

    def fail(self, reason: str) -> None:
        self.phase, self.reason = "failed", reason

    def _transition(self, phase: str, state: PushState) -> None:
        self.phase = phase
        self.phase_elapsed = self.contact_age = self.loss_age = self.settle_age = 0.0
        self._hand_start = {n: p.copy() for n, p in state.hands.items()}
        self._progress_time = self.elapsed
        self._progress_distance = self._distance_for_phase(state)
        if phase == "push":
            self._root_offset = state.qpos[:2] - state.body_position[:2]

    def _distance_for_phase(self, state: PushState) -> float:
        if self.phase == "approach":
            return float(np.linalg.norm(self.staging_point(state) - state.qpos[:2]))
        return self.remaining(state)

    def update(self, state: PushState, dt: float) -> bool:
        """Called after every physics control step; True interrupts the reference."""
        previous = self.phase
        if self.finished:
            return True
        self.elapsed += dt
        self.phase_elapsed += dt
        if not all(
            np.isfinite(a).all()
            for a in (
                state.qpos,
                state.body_position,
                state.body_rotation,
                state.body_velocity,
            )
        ):
            self.fail("Non-finite physical state")
        elif self.elapsed >= self.config.timeout:
            self.fail("Interaction time budget exhausted")
        if self.finished:
            return True

        touching = all(p.name in state.contacts for p in self.goal.points)
        self.contact_age = self.contact_age + dt if touching else 0.0
        self.loss_age = 0.0 if touching else self.loss_age + dt
        if self.phase == "approach":
            if self._distance_for_phase(state) <= 0.10:
                self._transition("contact", state)
        elif self.phase == "contact":
            if (
                not self.goal.maintain_contact
                and self.contact_age >= self.config.contact_dwell
            ):
                self._transition("push", state)
            elif not self.goal.maintain_contact and (
                self.phase_elapsed >= self.config.contact_windows * self.window_seconds
            ):
                self.fail("Contact not established within native contact windows")
        elif self.phase == "push":
            if self.remaining(state) <= 0.10:
                self._transition("settle", state)
            elif not self.attached and self.loss_age >= self.config.contact_loss:
                if self.retries >= self.config.reacquisitions:
                    self.fail("Contact reacquisition budget exhausted")
                else:
                    self.retries += 1
                    self._transition("contact", state)
        elif self.phase == "settle":
            contained = bool(
                np.all(
                    np.abs(state.footprint - np.asarray(self.goal.target_xy))
                    <= self.goal.goal_half_size
                )
            )
            settled = contained and np.linalg.norm(state.body_velocity) < 0.05
            self.settle_age = self.settle_age + dt if settled else 0.0
            if self.settle_age >= 0.5:
                self.phase, self.reason = "done", "Box footprint settled inside goal"
            elif self.phase_elapsed >= 3 * self.window_seconds:
                self.fail("Box did not settle inside goal")

        if self.phase in {"approach", "push"}:
            distance = self._distance_for_phase(state)
            if self._progress_distance - distance >= self.config.progress_distance:
                self._progress_time, self._progress_distance = self.elapsed, distance
            elif self.elapsed - self._progress_time >= self.config.stall_seconds:
                self.fail("No meaningful physical progress")
        return self.phase != previous

    def targets(
        self, state: PushState, frames: int, fps: float
    ) -> tuple[TimedTargets, ...]:
        root = state.qpos[:3]
        w, x, y, z = state.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        contacts = self.contact_points(state)
        direction = self._direction(state)
        samples = []
        indices = sorted(set(range(3, frames, 4)) | {frames - 1})
        for frame in indices:
            t = (frame + 1) / fps
            base = root.copy()
            hands: dict[str, np.ndarray] = {}
            if self.phase == "approach":
                delta = self.staging_point(state) - root[:2]
                distance = float(np.linalg.norm(delta))
                base[:2] += delta * min(
                    1.0, self.config.navigation_speed * t / max(distance, 1e-8)
                )
            elif self.phase == "contact":
                fraction = min(
                    1.0,
                    (self.phase_elapsed + t)
                    / (self.config.contact_windows * self.window_seconds),
                )
                for name, point in contacts.items():
                    # Millimetric bias, not the previous 20 cm penetration request.
                    target = point + np.r_[direction * 0.005, 0.0]
                    hands[name] = self._hand_start[name] + fraction * (
                        target - self._hand_start[name]
                    )
            else:
                advance = (
                    min(self.config.push_speed * t, self.remaining(state))
                    if self.phase == "push"
                    else 0.0
                )
                delta = np.r_[direction * advance, 0.0]
                if self.phase == "push":
                    base[:2] = state.body_position[:2] + self._root_offset + delta[:2]
                hands = {name: point + delta for name, point in contacts.items()}
            local_root = world_to_local @ (base - root)
            samples.append(
                TimedTargets(
                    frame,
                    (float(local_root[0]), float(local_root[1])),
                    tuple(
                        EndEffectorTarget(name, tuple(world_to_local @ (p - root)))
                        for name, p in hands.items()
                    ),
                )
            )
        return tuple(samples)
