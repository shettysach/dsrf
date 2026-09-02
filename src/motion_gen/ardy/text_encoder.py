from __future__ import annotations

from pathlib import Path

import torch
from ardy.model.llm2vec import LLM2VecEncoder

from shared.messages import ARDY_EMBEDDING_SIZE


class TextEncoder:
    """DSRF adapter for ARDY's checkpoint-compatible LLM2Vec encoder."""

    def __init__(self, model_path: Path, *, device: str) -> None:
        self._encoder = LLM2VecEncoder(
            model_path=model_path,
            device=device,
            llm_dim=ARDY_EMBEDDING_SIZE,
        )

    def encode(self, text: str) -> torch.Tensor:
        return self._encoder.encode(text)
