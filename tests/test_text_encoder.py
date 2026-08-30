import torch

import motion_gen.ardy.text_encoder as encoder_module
from motion_gen.ardy.text_encoder import TextEncoder


def test_text_encoder_delegates_to_ardy_with_selected_device(monkeypatch, tmp_path) -> None:
    received: dict[str, object] = {}

    class FakeLLM2VecEncoder:
        def __init__(self, **kwargs) -> None:
            received.update(kwargs)

        def encode(self, text: str) -> torch.Tensor:
            received["text"] = text
            return torch.arange(4096, dtype=torch.float64)

    monkeypatch.setattr(
        encoder_module,
        "LLM2VecEncoder",
        FakeLLM2VecEncoder,
    )

    encoder = TextEncoder(tmp_path, device="cuda:1")
    embedding = encoder.encode("walk forward")

    assert received == {
        "model_path": tmp_path,
        "device": "cuda:1",
        "llm_dim": 4096,
        "text": "walk forward",
    }
    assert embedding.shape == (4096,)
    assert embedding.dtype is torch.float64
