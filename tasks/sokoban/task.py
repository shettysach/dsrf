from __future__ import annotations

import os

from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec

_LEVEL = int(os.environ.get("SOKOBAN_LEVEL", "1"))


def _make_scene() -> SceneSpecFn:
    from tasks.sokoban.scene import make_sokoban_spec_fn

    return make_sokoban_spec_fn(level=_LEVEL)


def _robot_start() -> tuple[float, float, float]:
    from tasks.sokoban.scene import get_level, grid_to_world, level_positions

    return (*grid_to_world(level_positions(get_level(_LEVEL)).player), 0.76)


TASK = TaskSpec(
    name="sokoban",
    objective="Push every yellow box onto a separate green goal region.",
    make_scene=_make_scene,
    robot_initial_pos=_robot_start(),
    observation_camera=ObservationCameraSpec(
        world_position=(0.0, -7.8, 8.6),
        world_lookat=(0.0, 0.0, 0.0),
        fovy=58.0,
    ),
)
