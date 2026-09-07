"""Visual-only destination marker for the contact-free push motion."""

from __future__ import annotations

import mujoco

from tasks.push_motion.settings import PushMotionSettings
from tasks.spec import SceneSpecFn


def make_push_motion_spec_fn() -> SceneSpecFn:
    """Mark the base destination without introducing a collidable object."""

    def add_push_motion_goal(spec) -> None:
        settings = PushMotionSettings.from_env()
        spec.worldbody.add_geom(
            name="push_motion_goal",
            type=mujoco.mjtGeom.mjGEOM_BOX,  # ty: ignore[unresolved-attribute]
            pos=(settings.goal_x, 0.0, 0.01),
            size=(0.05, 0.75, 0.01),
            rgba=(0.1, 0.8, 0.2, 0.55),
            contype=0,
            conaffinity=0,
            mass=0.0,
        )

    return add_push_motion_goal
