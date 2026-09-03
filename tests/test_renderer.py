import torch

import sim.renderer as renderer_module
from sim.renderer import SimRenderer


def test_renderer_encodes_a_cpu_image(monkeypatch) -> None:
    renderer = object.__new__(SimRenderer)
    renderer.jpeg_quality = 85
    captured: dict[str, object] = {}

    def fake_encode(image: torch.Tensor, *, quality: int) -> torch.Tensor:
        captured["image"] = image
        captured["quality"] = quality
        return torch.tensor((255, 216, 255, 217), dtype=torch.uint8)

    monkeypatch.setattr(renderer_module, "encode_jpeg", fake_encode)

    encoded = renderer._encode(torch.zeros((2, 3, 3), dtype=torch.uint8))

    assert encoded == b"\xff\xd8\xff\xd9"
    assert captured["quality"] == 85
    image = captured["image"]
    assert isinstance(image, torch.Tensor)
    assert image.device.type == "cpu"
    assert image.shape == (3, 2, 3)
