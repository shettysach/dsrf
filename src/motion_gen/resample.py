from __future__ import annotations

import math

import numpy as np
import torch

from shared.messages import SONIC_FPS, EndEffectorTarget, MotionChunk


def _quat_slerp_batch(
    q1: torch.Tensor,
    q2: torch.Tensor,
    blend: torch.Tensor,
) -> torch.Tensor:
    """Batched shortest-path SLERP for quaternions in (w, x, y, z) order."""

    dot = torch.sum(q1 * q2, dim=-1)
    epsilon = torch.finfo(q1.dtype).eps * 4.0
    same_rotation = torch.abs(torch.abs(dot) - 1.0) < epsilon

    shortest_q2 = torch.where((dot < 0.0).unsqueeze(-1), -q2, q2)
    angle = torch.acos(torch.clamp(torch.abs(dot), -1.0, 1.0))
    same_rotation |= torch.abs(angle) < epsilon

    # The denominator is ignored for same-rotation rows, but keeping it non-zero
    # prevents NaNs before torch.where selects q1 for those rows.
    denominator = torch.where(same_rotation, torch.ones_like(angle), torch.sin(angle))
    q1_weight = torch.sin((1.0 - blend) * angle) / denominator
    q2_weight = torch.sin(blend * angle) / denominator
    interpolated = q1 * q1_weight.unsqueeze(-1) + shortest_q2 * q2_weight.unsqueeze(-1)
    interpolated = torch.where(same_rotation.unsqueeze(-1), q1, interpolated)

    # Match the scalar MJLab helper: exact endpoints return the original inputs,
    # including q2's sign at blend == 1.
    interpolated = torch.where((blend == 0.0).unsqueeze(-1), q1, interpolated)
    return torch.where((blend == 1.0).unsqueeze(-1), q2, interpolated)


def resample_qpos(source_qpos: torch.Tensor, *, source_fps: float) -> torch.Tensor:
    """Resample qpos on its current Torch device without host transfers."""

    output_frames = math.floor(source_qpos.shape[0] * SONIC_FPS / source_fps)
    source_positions = (
        torch.arange(
            output_frames,
            device=source_qpos.device,
            dtype=torch.float64,
        )
        * source_fps
        / SONIC_FPS
    )
    index_0 = torch.floor(source_positions).to(torch.long)
    index_0.clamp_(max=source_qpos.shape[0] - 1)
    index_1 = torch.clamp(index_0 + 1, max=source_qpos.shape[0] - 1)
    blend = (source_positions - index_0).to(dtype=source_qpos.dtype)

    qpos_0 = source_qpos.index_select(0, index_0)
    qpos_1 = source_qpos.index_select(0, index_1)
    output = torch.lerp(qpos_0, qpos_1, blend.unsqueeze(-1))
    output[:, 3:7] = _quat_slerp_batch(
        qpos_0[:, 3:7],
        qpos_1[:, 3:7],
        blend,
    )
    return output.contiguous()


def resample_motion(
    source_qpos: np.ndarray,
    *,
    source_fps: float,
    observation_id: int,
    command: str,
    reasoning: str | None = None,
    end_effectors: tuple[EndEffectorTarget, ...] = (),
) -> MotionChunk:
    """Convert NumPy backend output to a CPU MotionChunk at SONIC's frequency.

    Torch backends should call ``resample_qpos`` directly to remain on-device.
    """

    qpos = torch.as_tensor(source_qpos, dtype=torch.float32)
    output = resample_qpos(qpos, source_fps=source_fps)

    return MotionChunk(
        observation_id=observation_id,
        command=command,
        qpos=output.numpy(),
        reasoning=reasoning,
        end_effectors=end_effectors,
    )
