from __future__ import annotations

from pathlib import Path

import torch

from shared.onnx import StaticOnnxModel


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
        self.cuda_stream_ptr = None if cuda_stream is None else cuda_stream.cuda_stream
        self._model = StaticOnnxModel(
            model_path,
            device=device,
            cuda_stream=cuda_stream,
        )
        self.input = next(iter(self._model.inputs.values()))
        self.output = next(iter(self._model.outputs.values()))
        if self.input.shape != input_shape or self.output.shape != output_shape:
            raise ValueError(
                "Unexpected SONIC ONNX shapes: "
                f"input={tuple(self.input.shape)}, output={tuple(self.output.shape)}"
            )

    def run(self) -> torch.Tensor:
        self._model.run()
        return self.output
