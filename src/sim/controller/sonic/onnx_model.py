from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from shared.onnx import create_onnx_session


class OnnxModel:
    """ONNX model backed by stable Torch buffers and CUDA I/O binding."""

    def __init__(
        self,
        model_path: Path,
        *,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        device: torch.device,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.device = torch.device(device)

        self.cuda_stream_ptr = None if cuda_stream is None else cuda_stream.cuda_stream
        self.input = torch.zeros(input_shape, dtype=torch.float32, device=self.device)
        self.output = torch.empty(output_shape, dtype=torch.float32, device=self.device)
        self.session = create_onnx_session(
            model_path,
            device=self.device,
            cuda_stream=cuda_stream,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._binding = self._create_binding()

    def _create_binding(self):
        device_id = 0
        if self.device.type == "cuda":
            device_id = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )

        binding = self.session.io_binding()
        binding.bind_input(
            self.input_name,
            self.device.type,
            device_id,
            np.float32,
            self.input.shape,
            self.input.data_ptr(),
        )
        binding.bind_output(
            self.output_name,
            self.device.type,
            device_id,
            np.float32,
            self.output.shape,
            self.output.data_ptr(),
        )
        return binding

    def run(self) -> torch.Tensor:
        self.session.run_with_iobinding(self._binding)
        return self.output
