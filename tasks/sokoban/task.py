from __future__ import annotations

import os

from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec

_LEVEL = int(os.environ.get("SOKOBAN_LEVEL", "1"))
# The default G1 faces +X (screen-right in the overhead board view).  A +90°
# yaw makes it face +Y, into the board away from the south-side camera.
_FACING_BOARD_QUAT = (0.7071067811865476, 0.0, 0.0, 0.7071067811865475)


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
    robot_initial_rot=_FACING_BOARD_QUAT,
    observation_camera=ObservationCameraSpec(
        # Translate with the robot while retaining this world orientation. This
        # keeps the screen axes stable when the robot turns and provides a
        # closer, less top-down view than the original fixed overview.
        world_position=(0.5, -4.5, 4.2),
        world_lookat=(0.5, 0.5, 0.0),
        follow_robot_translation=True,
        fovy=62.0,
    ),
)
