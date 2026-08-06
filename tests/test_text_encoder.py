from types import SimpleNamespace

import numpy as np
import torch

from encoder.encoder import TextEncoder


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
    assert embedding.dtype == np.float32
    np.testing.assert_array_equal(embedding, np.full(4096, 3.0, dtype=np.float32))
