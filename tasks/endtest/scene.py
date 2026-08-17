from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

from tasks.spec import SceneSpecFn

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]

TARGET_CENTER_X = 0.55
TARGET_CENTER_Y = 0.22
TARGET_HALF_SIZE = (0.18, 0.14, 0.01)
_TARGET_RGBA = (0.1, 0.85, 0.25, 1.0)


def make_endtest_spec_fn() -> SceneSpecFn:
    """Create two visual-only green foot-placement targets ahead of the robot."""

    def add_endtest(spec: MjSpec) -> None:
        _add_target(spec, "left", TARGET_CENTER_Y)
        _add_target(spec, "right", -TARGET_CENTER_Y)

    return add_endtest


def _add_target(spec: "MjSpec", side: str, y: float) -> None:
    body = spec.worldbody.add_body(name=f"endtest_{side}_target")
    body.pos = (TARGET_CENTER_X, y, TARGET_HALF_SIZE[2])
    body.add_geom(
        name=f"endtest_{side}_target_surface",
        type=MJGEOM_BOX,
        size=TARGET_HALF_SIZE,
        rgba=_TARGET_RGBA,
        # These are visual placement targets, not obstacles to step over.
        contype=0,
        conaffinity=0,
    )
