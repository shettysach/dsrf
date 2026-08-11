import math

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_slerp

from motion_gen.resample import resample_motion, resample_qpos


def _scalar_reference(qpos: torch.Tensor, source_fps: float) -> torch.Tensor:
    output_frames = math.floor(qpos.shape[0] * 50 / source_fps)
    output = torch.empty(
        (output_frames, qpos.shape[1]), dtype=qpos.dtype, device=qpos.device
    )
    for output_index in range(output_frames):
        source_position = output_index * source_fps / 50
        index_0 = min(math.floor(source_position), qpos.shape[0] - 1)
        index_1 = min(index_0 + 1, qpos.shape[0] - 1)
        blend = float(source_position - index_0)
        output[output_index] = torch.lerp(qpos[index_0], qpos[index_1], blend)
        output[output_index, 3:7] = quat_slerp(
            qpos[index_0, 3:7],
            qpos[index_1, 3:7].clone(),
            blend,
        )
    return output


def test_resample_64_frames_to_106_without_mutating_input() -> None:
    qpos = np.zeros((64, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[1::2, 3] = -1.0
    qpos[:, 7] = np.arange(64)
    original = qpos.copy()

    chunk = resample_motion(
        qpos,
        source_fps=30,
        observation_id=2,
        command="walk forward",
    )

    assert chunk.qpos.shape == (106, 36)
    assert chunk.observation_id == 2
    assert chunk.command == "walk forward"
    np.testing.assert_array_equal(qpos, original)
    np.testing.assert_allclose(np.linalg.norm(chunk.qpos[:, 3:7], axis=1), 1.0)


def test_resample_25_fps_backend_to_sonic_fps() -> None:
    qpos = np.zeros((25, 36), dtype=np.float32)
    qpos[:, 3] = 1.0

    chunk = resample_motion(
        qpos,
        source_fps=25,
        observation_id=3,
        command="walk forward",
    )

    assert chunk.qpos.shape == (50, 36)


def test_vectorized_resampling_matches_scalar_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    qpos = torch.randn((17, 36), generator=generator)
    qpos[:, 3:7] = torch.nn.functional.normalize(qpos[:, 3:7], dim=-1)
    original = qpos.clone()

    actual = resample_qpos(qpos, source_fps=30)
    expected = _scalar_reference(qpos, source_fps=30)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(qpos, original)


def test_resample_qpos_preserves_torch_dtype_and_device() -> None:
    qpos = torch.zeros((2, 36), dtype=torch.float64)
    qpos[:, 3] = 1.0
    qpos[1, 3] = math.cos(math.pi / 4)
    qpos[1, 6] = math.sin(math.pi / 4)

    output = resample_qpos(qpos, source_fps=25)

    assert output.dtype == qpos.dtype
    assert output.device == qpos.device
    assert output.is_contiguous()
    expected_halfway = torch.tensor(
        [math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)],
        dtype=qpos.dtype,
    )
    torch.testing.assert_close(output[1, 3:7], expected_halfway)
