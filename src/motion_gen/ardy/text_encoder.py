from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from shared.messages import ARDY_EMBEDDING_SIZE


class TextEncoder:
    """Encode commands with a local Transformers model and masked mean pooling."""

    def __init__(self, model_path: Path, *, device: str) -> None:
        self.device = torch.device(device)
        self.tokenizer: Any = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModel.from_pretrained(
            str(model_path),
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self.model(**inputs)

        hidden = output.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        if embedding.shape != (1, ARDY_EMBEDDING_SIZE):
            raise ValueError(
                "Text encoder must produce shape "
                f"[1, {ARDY_EMBEDDING_SIZE}], got {tuple(embedding.shape)}"
            )
        return embedding[0].float().contiguous()
