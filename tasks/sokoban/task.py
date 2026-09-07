from __future__ import annotations

import os

from tasks.spec import ObservationCameraSpec, SceneSpecFn, TaskSpec

_LEVEL = int(os.environ.get("SOKOBAN_LEVEL", "1"))
# The default G1 faces +X (screen-right in the overhead board view).  A +90°
# yaw makes it face +Y, into the board away from its trailing camera.
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
        # This is intentionally torso-relative rather than a world camera.
        # Keeping the robot near the centre makes its local walk directions
        # visible, and the tighter framing gives boxes and walls enough pixels
        # to distinguish reliably.  The steep elevation retains board context.
        distance=5.25,
        elevation=-65.0,
        fovy=62.0,
    ),
)
