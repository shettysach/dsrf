from types import SimpleNamespace

import torch

import motion_gen.ardy.text_encoder as encoder_module
from motion_gen.ardy.text_encoder import TextEncoder


def test_text_encoder_loads_model_on_selected_device(monkeypatch, tmp_path) -> None:
    received: dict[str, object] = {}
    model = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(
        encoder_module.AutoTokenizer,
        "from_pretrained",
        lambda path: object(),
    )
    monkeypatch.setattr(
        encoder_module.AutoModel,
        "from_pretrained",
        lambda path, **kwargs: received.update(kwargs) or model,
    )

    encoder = TextEncoder(tmp_path, device="cuda:1")

    assert encoder.device == torch.device("cuda:1")
    assert received["device_map"] == "cuda:1"


def test_text_encoder_returns_masked_mean_float32_embedding() -> None:
    encoder = TextEncoder.__new__(TextEncoder)
    encoder.device = torch.device("cpu")
    encoder.tokenizer = lambda *args, **kwargs: {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    hidden = torch.zeros((1, 3, 4096), dtype=torch.float64)
    hidden[:, 0].fill_(2.0)
    hidden[:, 1].fill_(4.0)
    hidden[:, 2].fill_(100.0)
    encoder.model = lambda **kwargs: SimpleNamespace(last_hidden_state=hidden)

    embedding = encoder.encode("walk forward")

    assert embedding.shape == (4096,)
    assert embedding.dtype is torch.float32
    assert embedding.device == torch.device("cpu")
    torch.testing.assert_close(embedding, torch.full((4096,), 3.0))
