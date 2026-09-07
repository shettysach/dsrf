from __future__ import annotations

from tasks.push_motion.settings import PushMotionSettings
from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec


def _make_scene() -> SceneSpecFn:
    from tasks.push_motion.scene import make_push_motion_spec_fn

    return make_push_motion_spec_fn()


_SETTINGS = PushMotionSettings.from_env()


TASK = TaskSpec(
    name="push_motion",
    objective="Approach the push pose, extend both palms forward, and walk the push motion to the green goal.",
    make_scene=_make_scene,
    robot_initial_pos=(0.0, 0.0, 0.76),
    observation_camera=ObservationCameraSpec(
        world_position=(0.0, -6.5, 4.5),
        world_lookat=(_SETTINGS.goal_x / 2.0, 0.0, 0.8),
        fovy=65.0,
    ),
)
