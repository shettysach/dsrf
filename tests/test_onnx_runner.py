from types import SimpleNamespace
from typing import Any, cast

import torch

from controller.sonic.onnx_model import OnnxModel


class _Binding:
    def __init__(self) -> None:
        self.input_device_id: int | None = None
        self.output_device_id: int | None = None

    def bind_input(self, *args) -> None:
        self.input_device_id = args[2]

    def bind_output(self, *args) -> None:
        self.output_device_id = args[2]


def test_io_binding_uses_cuda_device_index() -> None:
    binding = _Binding()
    model = OnnxModel.__new__(OnnxModel)
    model.session = cast(Any, SimpleNamespace(io_binding=lambda: binding))
    model.input_name = "input"
    model.output_name = "output"
    model.device = torch.device("cuda:1")
    cast(Any, model).device_id = 1
    model.input = torch.zeros((1, 2))
    model.output = torch.empty((1, 3))

    model._create_binding()

    assert binding.input_device_id == 1
    assert binding.output_device_id == 1
