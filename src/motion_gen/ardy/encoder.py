from __future__ import annotations

import torch

from shared.messages import ARDY_EMBEDDING_SIZE


def prepare_conditioning(
    embedding: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embedding.shape != (ARDY_EMBEDDING_SIZE,):
        raise ValueError(
            "ARDY embedding must have shape "
            f"[{ARDY_EMBEDDING_SIZE}], got {tuple(embedding.shape)}"
        )
    text_feat = embedding.detach().to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    text_feat = text_feat.reshape(1, 1, ARDY_EMBEDDING_SIZE)
    text_pad_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
    return text_feat, text_pad_mask
