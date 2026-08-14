from __future__ import annotations

import torch
from ardy.motion_rep.tools import RotateFeatures


def build_waypoint_constraints(
    motion_rep,
    root_history: torch.Tensor,
    root_heading: torch.Tensor,
    target_xys: tuple[tuple[float, float], ...],
    *,
    generated_frames: int,
    history_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert robot-local targets into evenly spaced ARDY root endpoints."""
    if (
        root_history.ndim != 2
        or root_history.shape[0] < 2
        or root_history.shape[1] != 3
    ):
        raise ValueError(
            "ARDY root history must have shape [T >= 2, 3], "
            f"got {tuple(root_history.shape)}"
        )

    current_root_2d = root_history[-1, [0, 2]]
    if root_heading.numel() != 1 or not torch.isfinite(root_heading).all():
        raise ValueError("ARDY root heading must be one finite angle")
    if not target_xys:
        raise ValueError("ARDY waypoint constraints require at least one target")
    local_ardy = torch.tensor(
        [[left, forward] for forward, left in target_xys],
        dtype=current_root_2d.dtype,
        device=device,
    )
    heading = root_heading.reshape(()).to(dtype=current_root_2d.dtype, device=device)
    delta_ardy = (
        RotateFeatures(heading.unsqueeze(0))
        .rotate_2d_positions(local_ardy.unsqueeze(0))
        .squeeze(0)
    )
    relative_indices = (
        torch.arange(1, len(target_xys) + 1, device=device)
        * generated_frames
        // len(target_xys)
    )
    root_2d = current_root_2d + delta_ardy
    frame_indices = relative_indices + history_frames - 1
    observed_motion, motion_mask = motion_rep.create_conditions(
        {"root_2d": [frame_indices]},
        {"root_2d": [root_2d]},
        generated_frames + history_frames,
        True,
        device,
    )
    return motion_mask.unsqueeze(0), observed_motion.unsqueeze(0)
