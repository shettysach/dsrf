"""Runtime capture and toggling of predeclared MJWarp weld constraints."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array((quat[0], -quat[1], -quat[2], -quat[3]))


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


class HandBoxWelds:
    """Two initially inactive hand-to-box welds for the scripted benchmark."""

    def __init__(self, simulation, body: str, hands: tuple[str, ...]) -> None:
        if body != "box":
            raise ValueError("The scripted weld capability supports only the box body")
        if set(hands) != {"left_hand", "right_hand"}:
            raise ValueError("The scripted weld capability requires both hands")
        self._sim = simulation.mjlab_env.sim
        self._model = self._sim.mj_model
        self._ids = {
            hand: int(self._model.equality(f"{hand}_box_weld").id) for hand in hands
        }
        self._hand_geoms = {hand: simulation._hand_geom_ids[hand] for hand in hands}
        self._box_body = int(self._model.body("box/box").id)
        for hand, weld_id in self._ids.items():
            hand_body = int(self._model.geom_bodyid[self._hand_geoms[hand]])
            if int(self._model.eq_obj1id[weld_id]) != hand_body:
                raise ValueError(f"{hand} weld does not target its collision body")
            if int(self._model.eq_obj2id[weld_id]) != self._box_body:
                raise ValueError(f"{hand} weld does not target box/box")
        self.attached = False

    def attach(self) -> None:
        """Capture current transforms, then enable both constraints in place."""
        if self.attached:
            return
        data = self._sim.data
        values = self._sim.wp_model.eq_data.numpy()
        box_position = _host(data.xpos[0, self._box_body])
        box_quat = _host(data.xquat[0, self._box_body])
        box_rotation = _rotation_matrix(box_quat)
        for hand, weld_id in self._ids.items():
            hand_body = int(self._model.geom_bodyid[self._hand_geoms[hand]])
            point = _host(data.geom_xpos[0, self._hand_geoms[hand]])
            hand_position = _host(data.xpos[0, hand_body])
            hand_quat = _host(data.xquat[0, hand_body])
            hand_rotation = _rotation_matrix(hand_quat)
            # MJWarp body-body layout: box anchor, hand anchor,
            # inv(hand_quat) * box_quat, torque scale.
            values[0, weld_id] = np.concatenate(
                (
                    box_rotation.T @ (point - box_position),
                    hand_rotation.T @ (point - hand_position),
                    _quat_multiply(_quat_conjugate(hand_quat), box_quat),
                    np.array((1.0,)),
                )
            )
        raw = self._sim.wp_model.eq_data
        wp.copy(raw, wp.array(values, dtype=raw.dtype, device=raw.device))
        for weld_id in self._ids.values():
            data.eq_active[0, weld_id] = True
        self.attached = True

    def detach(self) -> None:
        if not self.attached:
            return
        for weld_id in self._ids.values():
            self._sim.data.eq_active[0, weld_id] = False
        self.attached = False


def _host(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64, copy=True)
