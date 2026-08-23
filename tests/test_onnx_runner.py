from types import SimpleNamespace

import numpy as np
import torch

from shared import onnx as onnx_utils


class _Binding:
    def __init__(self) -> None:
        self.input_device_id: int | None = None
        self.output_device_id: int | None = None

    def bind_input(self, *args) -> None:
        self.input_device_id = args[2]

    def bind_output(self, *args) -> None:
        self.output_device_id = args[2]


class _Session:
    def __init__(self, binding: _Binding) -> None:
        self._binding = binding

    def get_inputs(self):
        return [SimpleNamespace(name="input", type="tensor(float)", shape=[1, 2])]

    def get_outputs(self):
        return [SimpleNamespace(name="output", type="tensor(float)", shape=[1, 3])]

    def io_binding(self):
        return self._binding

    def run_with_iobinding(self, binding) -> None:
        assert binding is self._binding


def test_io_binding_uses_cuda_device_index(monkeypatch) -> None:
    binding = _Binding()
    monkeypatch.setattr(
        onnx_utils,
        "create_onnx_session",
        lambda *args, **kwargs: _Session(binding),
    )
    monkeypatch.setattr(
        onnx_utils,
        "_allocate_tensor",
        lambda value, device: torch.empty(
            tuple(value.shape), dtype=torch.float32, device="cpu"
        ),
    )

    model = onnx_utils.StaticOnnxModel(
        SimpleNamespace(),  # type: ignore[arg-type]
        device=torch.device("cuda:1"),
    )

    assert binding.input_device_id == 1
    assert binding.output_device_id == 1
    assert model.inputs["input"].shape == (1, 2)


def test_static_onnx_model_reuses_bound_buffers(monkeypatch) -> None:
    binding = _Binding()
    session = _Session(binding)
    monkeypatch.setattr(
        onnx_utils,
        "create_onnx_session",
        lambda *args, **kwargs: session,
    )
    model = onnx_utils.StaticOnnxModel(
        SimpleNamespace(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    outputs = model.run({"input": torch.tensor([[1.0, 2.0]])})

    np.testing.assert_array_equal(model.inputs["input"].numpy(), [[1.0, 2.0]])
    assert outputs["output"] is model.outputs["output"]
