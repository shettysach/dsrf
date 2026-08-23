from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from sim.controller.sonic.policy import MotionReference

REFERENCE_GHOST_COLOR = np.array((0.5, 0.7, 0.5, 0.5), dtype=np.float32)


class ReferenceGhost:
    """Draw the active controller reference through MJLab's debug visualizer."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        reference: MotionReference,
    ) -> None:
        self._env = env
        self._robot = env.scene["robot"]
        self._reference = reference
        self._ghost_model = self._make_ghost_model()  # mujoco.MjModel

    def draw(self, visualizer: DebugVisualizer) -> None:
        pose = self._reference.visualization_pose()
        if pose is None:
            return

        root_pos_w, root_quat_w, joint_pos = pose
        indexing = self._robot.indexing
        free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
        joint_q_adr = indexing.joint_q_adr.cpu().numpy()

        qpos = np.zeros(self._env.sim.mj_model.nq, dtype=np.float64)
        qpos[free_joint_q_adr[:3]] = root_pos_w.detach().cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = root_quat_w.detach().cpu().numpy()
        qpos[joint_q_adr] = joint_pos.detach().cpu().numpy()
        visualizer.add_ghost_mesh(
            qpos,
            model=self._ghost_model,
            alpha=float(REFERENCE_GHOST_COLOR[3]),
            label="sonic_reference",
        )

    def _make_ghost_model(self) -> Any:  # mujoco.MjModel
        model = copy.deepcopy(self._env.sim.mj_model)
        collision_geoms = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_rgba[collision_geoms, 3] = 0.0
        model.geom_rgba[~collision_geoms] = REFERENCE_GHOST_COLOR
        return model
