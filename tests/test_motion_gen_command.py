from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from motion_gen.kinematic_planner import (
    KinematicPlanner,
    PlannerMode,
    planner_direction,
    planner_mode,
)


def test_navigation_modes_are_intentionally_small() -> None:
    assert planner_mode("stand") is PlannerMode.IDLE
    assert planner_mode("walk") is PlannerMode.WALK
    with pytest.raises(ValueError, match="Unsupported motion"):
        planner_mode("run")


def test_navigation_directions_map_to_local_vectors() -> None:
    assert planner_direction("forward") == (1.0, 0.0)
    assert planner_direction("left") == (0.0, 1.0)
    with pytest.raises(ValueError, match="Unsupported direction"):
        planner_direction("north")


def test_kinematic_planner_visits_waypoints_in_order() -> None:
    captured: list[dict[str, np.ndarray]] = []

    def run(_outputs, inputs):
        captured.append(inputs)
        qpos = np.tile(_standing(), (1, 8, 1))
        qpos[0, :, 0] = inputs["specific_target_positions"][0, 0, 0]
        qpos[0, :, 1] = inputs["specific_target_positions"][0, 0, 1]
        return qpos, np.array([8], dtype=np.int64)

    planner = KinematicPlanner.__new__(KinematicPlanner)
    planner.session = cast(Any, SimpleNamespace(run=run))
    planner._context = np.tile(_standing(), (1, 4, 1))

    qpos = planner.generate("walk", ((1.0, 0.5), (2.0, -0.5)))

    assert len(captured) == 2
    np.testing.assert_allclose(
        captured[0]["specific_target_positions"][0, :, :2], [[1.0, 0.5]] * 4
    )
    np.testing.assert_allclose(
        captured[1]["specific_target_positions"][0, :, :2], [[2.0, -0.5]] * 4
    )
    assert qpos.shape == (16, 36)


def test_local_lateral_target_uses_world_target_and_preserves_heading() -> None:
    captured: dict[str, np.ndarray] = {}

    def run(_outputs, inputs):
        captured.update(inputs)
        qpos = np.tile(_standing(), (1, 8, 1))
        return qpos, np.array([8], dtype=np.int64)

    planner = KinematicPlanner.__new__(KinematicPlanner)
    planner.session = cast(Any, SimpleNamespace(run=run))
    planner._context = np.tile(_standing(), (1, 4, 1))
    planner.generate("walk", ((1.0, 0.5),))

    np.testing.assert_allclose(
        captured["specific_target_positions"][0, :, :2],
        np.array([[1.0, 0.5]] * 4),
    )
    np.testing.assert_allclose(captured["specific_target_headings"], 0.0)
    np.testing.assert_allclose(captured["facing_direction"], [[1.0, 0.0, 0.0]])
    assert captured["has_specific_target"].tolist() == [[1]]


def test_stand_has_no_specific_target() -> None:
    captured: dict[str, np.ndarray] = {}

    def run(_outputs, inputs):
        captured.update(inputs)
        qpos = np.tile(_standing(), (1, 8, 1))
        return qpos, np.array([8], dtype=np.int64)

    planner = KinematicPlanner.__new__(KinematicPlanner)
    planner.session = cast(Any, SimpleNamespace(run=run))
    planner._context = np.tile(_standing(), (1, 4, 1))
    planner.generate("stand", ())

    assert captured["has_specific_target"].tolist() == [[0]]
    np.testing.assert_allclose(captured["movement_direction"], 0.0)


def test_stand_direction_can_cue_a_turn() -> None:
    captured: dict[str, np.ndarray] = {}

    def run(_outputs, inputs):
        captured.update(inputs)
        qpos = np.tile(_standing(), (1, 8, 1))
        return qpos, np.array([8], dtype=np.int64)

    planner = KinematicPlanner.__new__(KinematicPlanner)
    planner.session = cast(Any, SimpleNamespace(run=run))
    planner._context = np.tile(_standing(), (1, 4, 1))
    planner.generate("stand", (), "left")

    assert captured["has_specific_target"].tolist() == [[0]]
    np.testing.assert_allclose(captured["movement_direction"], [[0.0, 1.0, 0.0]])


def test_direction_is_robot_relative() -> None:
    captured: dict[str, np.ndarray] = {}

    def run(_outputs, inputs):
        captured.update(inputs)
        qpos = np.tile(_standing(), (1, 8, 1))
        return qpos, np.array([8], dtype=np.int64)

    planner = KinematicPlanner.__new__(KinematicPlanner)
    planner.session = cast(Any, SimpleNamespace(run=run))
    context_frame = _standing()
    yaw = np.pi / 2.0
    context_frame[3] = np.cos(yaw / 2.0)
    context_frame[6] = np.sin(yaw / 2.0)
    planner._context = np.tile(context_frame, (1, 4, 1))

    planner.generate("walk", (), "forward")

    np.testing.assert_allclose(
        captured["movement_direction"], [[0.0, 1.0, 0.0]], atol=1e-6
    )
    np.testing.assert_allclose(
        captured["facing_direction"], [[0.0, 1.0, 0.0]], atol=1e-6
    )


def _standing() -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[3] = 1.0
    return qpos
