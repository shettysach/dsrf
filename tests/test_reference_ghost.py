from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from shared.messages import MotionChunk
from sim.reference_ghost import ReferenceGhost
from tracker.reference import MotionReference


def _motion() -> MotionChunk:
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[0, :3] = [1.0, 2.0, 0.8]
    qpos[1, :3] = [2.0, 2.0, 0.8]
    qpos[1, 7:] = np.arange(29)
    return MotionChunk(0, "walk forward", qpos)


def test_reference_visualization_pose_is_aligned_and_advances() -> None:
    reference = MotionReference(torch.device("cpu"))
    half_sqrt = 2**-0.5
    reference.load(
        _motion(),
        robot_pos_w=torch.tensor([10.0, 20.0, 0.8]),
        robot_quat_w=torch.tensor([half_sqrt, 0.0, 0.0, half_sqrt]),
    )

    initial = reference.visualization_pose()
    assert initial is not None
    np.testing.assert_allclose(initial[0].numpy(), [10.0, 20.0, 0.8])

    assert not reference.advance()
    current = reference.visualization_pose()
    assert current is not None
    np.testing.assert_allclose(current[0].numpy(), [10.0, 21.0, 0.8], atol=1e-6)
    np.testing.assert_allclose(
        current[1].numpy(), [half_sqrt, 0.0, 0.0, half_sqrt], atol=1e-6
    )
    np.testing.assert_array_equal(current[2].numpy(), np.arange(29))

    assert reference.advance()
    assert reference.visualization_pose() is None


def test_reference_preserves_generated_relative_root_z() -> None:
    motion = _motion()
    motion.qpos[0, 2] = 0.5
    motion.qpos[1, 2] = 0.45
    reference = MotionReference(torch.device("cpu"))
    reference.load(
        motion,
        robot_pos_w=torch.tensor([10.0, 20.0, 0.8]),
        robot_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )

    initial = reference.visualization_pose()
    assert initial is not None
    np.testing.assert_allclose(initial[0].numpy(), [10.0, 20.0, 0.8])

    assert not reference.advance()
    current = reference.visualization_pose()
    assert current is not None
    np.testing.assert_allclose(current[0].numpy(), [11.0, 20.0, 0.75])


class _RecordingVisualizer:
    def __init__(self) -> None:
        self.ghosts: list[dict[str, Any]] = []

    def add_ghost_mesh(self, qpos, **kwargs) -> None:
        self.ghosts.append({"qpos": qpos, **kwargs})


def test_reference_ghost_builds_model_qpos_and_hides_collision_geoms() -> None:
    reference = MotionReference(torch.device("cpu"))
    reference.load(
        _motion(),
        robot_pos_w=torch.tensor([3.0, 4.0, 0.8]),
        robot_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    model = SimpleNamespace(
        nq=36,
        ngeom=2,
        geom_contype=np.array([0, 1]),
        geom_conaffinity=np.array([0, 0]),
        geom_rgba=np.ones((2, 4), dtype=np.float32),
    )
    indexing = SimpleNamespace(
        free_joint_q_adr=torch.arange(7),
        joint_q_adr=torch.arange(7, 36),
    )
    env = SimpleNamespace(
        sim=SimpleNamespace(mj_model=model),
        scene={"robot": SimpleNamespace(indexing=indexing)},
    )
    visualizer = _RecordingVisualizer()

    ghost = ReferenceGhost(cast(Any, env), reference)
    ghost.draw(cast(Any, visualizer))

    assert len(visualizer.ghosts) == 1
    rendered = visualizer.ghosts[0]
    np.testing.assert_allclose(rendered["qpos"][:3], [3.0, 4.0, 0.8])
    np.testing.assert_allclose(rendered["model"].geom_rgba[0], [0.5, 0.7, 0.5, 0.5])
    assert rendered["model"].geom_rgba[1, 3] == 0.0
    assert rendered["label"] == "sonic_reference"
