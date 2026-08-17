from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sim.env import MjlabEnv

_SMALL_BLOCK_BODY = "stairs_small_block"
_SMALL_BLOCK_REST_WELD = "stairs_small_block_rest_weld"
_SMALL_BLOCK_HALF_SIZE = (0.25, 0.45, 0.15)
_CONTACT_PADDING_M = 0.04


class StairsGrasp:
    """Attach the small stairs block to the first palm that reaches it.

    The scene holds the block fixed with a world weld until a palm reaches its
    surface. This task-local interaction substitutes for G1 finger actuation,
    which the SONIC control model does not expose.
    """

    def __init__(self, simulation: "MjlabEnv") -> None:
        self._simulation = simulation
        sim = simulation.unwrapped.sim
        model = sim.mj_model
        self._sim = sim
        self._block_body_id = model.body(_SMALL_BLOCK_BODY).id
        self._rest_weld_id = model.equality(_SMALL_BLOCK_REST_WELD).id
        self._palms = tuple(
            (
                model.site(f"robot/{side}_palm").id,
                model.body(f"robot/{side}_wrist_yaw_link").id,
                model.equality(f"stairs_small_block_{side}_hand_weld").id,
            )
            for side in ("left", "right")
        )
        self._attached = False

    def before_step(self) -> None:
        if self._attached:
            return
        with self._simulation.compute_context():
            for palm_site_id, hand_body_id, hand_weld_id in self._palms:
                if self._palm_reaches_block(palm_site_id):
                    self._attach(hand_body_id, hand_weld_id)
                    self._attached = True
                    return

    def _palm_reaches_block(self, palm_site_id: int) -> bool:
        data = self._sim.data
        block_pos = data.xpos[0, self._block_body_id]
        block_rotation = data.xmat[0, self._block_body_id].reshape(3, 3)
        palm_pos = data.site_xpos[0, palm_site_id]
        local_palm_pos = block_rotation.T @ (palm_pos - block_pos)
        half_size = torch.as_tensor(
            _SMALL_BLOCK_HALF_SIZE,
            dtype=local_palm_pos.dtype,
            device=local_palm_pos.device,
        )
        closest = torch.clamp(local_palm_pos, min=-half_size, max=half_size)
        distance = torch.linalg.vector_norm(local_palm_pos - closest)
        return bool(distance <= _CONTACT_PADDING_M)

    def _attach(self, hand_body_id: int, hand_weld_id: int) -> None:
        data = self._sim.data
        model = self._sim.model
        hand_pos = data.xpos[0, hand_body_id]
        hand_quat = data.xquat[0, hand_body_id]
        hand_rotation = data.xmat[0, hand_body_id].reshape(3, 3)
        block_pos = data.xpos[0, self._block_body_id]
        block_quat = data.xquat[0, self._block_body_id]

        # Set the disabled weld's anchors and relative orientation to the
        # current pose, so enabling it does not snap the block into a reference
        # configuration from scene construction.
        model.eq_data[hand_weld_id, :3] = 0.0
        model.eq_data[hand_weld_id, 3:6] = hand_rotation.T @ (block_pos - hand_pos)
        model.eq_data[hand_weld_id, 6:10] = _relative_quaternion(
            hand_quat, block_quat
        )
        data.eq_active[0, self._rest_weld_id] = False
        data.eq_active[0, hand_weld_id] = True


def _relative_quaternion(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Return second's wxyz orientation in first's frame."""

    first_conjugate = first * torch.tensor(
        (1.0, -1.0, -1.0, -1.0), dtype=first.dtype, device=first.device
    )
    w1, x1, y1, z1 = first_conjugate
    w2, x2, y2, z2 = second
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )
