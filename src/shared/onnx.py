from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    import torch


_TORCH_DTYPES = {
    "tensor(float)": "float32",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
}

_NUMPY_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


def create_onnx_session(
    model_path: Path,
    *,
    device: torch.device,
    cuda_stream: torch.cuda.Stream | None = None,
) -> ort.InferenceSession:
    import torch

    if device.type == "cpu":
        return ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    ort.preload_dlls()

    device_id = torch.cuda.current_device() if device.index is None else device.index
    provider_options = {"device_id": str(device_id)}

    if cuda_stream is not None:
        stream_device_id = cuda_stream.device.index
        assert stream_device_id == device_id
        provider_options["user_compute_stream"] = str(cuda_stream.cuda_stream)

    return ort.InferenceSession(
        model_path,
        sess_options=ort.SessionOptions(),
        providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"],
    )


class StaticOnnxModel:
    """Static-shape ONNX model with stable Torch buffers and I/O binding."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: torch.device,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self.session = create_onnx_session(
            model_path,
            device=self.device,
            cuda_stream=cuda_stream,
        )
        self.inputs = {
            value.name: _allocate_tensor(value, self.device)
            for value in self.session.get_inputs()
        }
        for tensor in self.inputs.values():
            tensor.zero_()
        self.outputs = {
            value.name: _allocate_tensor(value, self.device)
            for value in self.session.get_outputs()
        }
        self._binding = self.session.io_binding()
        device_id = _device_id(self.device)
        for value in self.session.get_inputs():
            tensor = self.inputs[value.name]
            self._binding.bind_input(
                value.name,
                self.device.type,
                device_id,
                _NUMPY_DTYPES[value.type],
                tensor.shape,
                tensor.data_ptr(),
            )
        for value in self.session.get_outputs():
            tensor = self.outputs[value.name]
            self._binding.bind_output(
                value.name,
                self.device.type,
                device_id,
                _NUMPY_DTYPES[value.type],
                tensor.shape,
                tensor.data_ptr(),
            )

    def run(
        self,
        inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs is not None:
            for name, value in inputs.items():
                self.inputs[name].copy_(value)
        self.session.run_with_iobinding(self._binding)
        return self.outputs


def _allocate_tensor(value: ort.NodeArg, device: torch.device) -> torch.Tensor:
    import torch

    if value.type not in _TORCH_DTYPES:
        raise TypeError(f"Unsupported ONNX tensor type {value.type!r} for {value.name}")
    if any(not isinstance(size, int) for size in value.shape):
        raise ValueError(f"ONNX tensor {value.name!r} must have a static shape")
    dtype = getattr(torch, _TORCH_DTYPES[value.type])
    return torch.empty(tuple(value.shape), dtype=dtype, device=device)


def _device_id(device: torch.device) -> int:
    import torch

    if device.type != "cuda":
        return 0
    return device.index if device.index is not None else torch.cuda.current_device()
