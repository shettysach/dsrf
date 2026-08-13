import math
from types import SimpleNamespace

import pytest
import torch

from motion_gen.ardy.constraints import build_waypoint_constraints


def _conditions():
    received: dict[str, object] = {}

    def create_conditions(index, data, length, normalize, device):
        received.update(
            index=index,
            data=data,
            length=length,
            normalize=normalize,
            device=device,
        )
        return torch.zeros((129, 8)), torch.zeros((129, 8))

    return SimpleNamespace(create_conditions=create_conditions), received


def test_local_waypoint_becomes_ardy_root_endpoint() -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])

    motion_mask, observed_motion = build_waypoint_constraints(
        motion_rep,
        root_history,
        torch.tensor(0.0),
        (0.8, 0.3),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    assert motion_mask.shape == (1, 129, 8)
    assert observed_motion.shape == (1, 129, 8)
    assert received["index"]["root_2d"][0].tolist() == [128]
    torch.testing.assert_close(
        received["data"]["root_2d"][0],
        torch.tensor([[1.3, 2.8]]),
    )


@pytest.mark.parametrize(
    ("heading", "expected_endpoint"),
    [
        (0.0, [1.0, 3.0]),
        (math.pi / 2.0, [0.0, 2.0]),
        (math.pi, [1.0, 1.0]),
        (-math.pi / 2.0, [2.0, 2.0]),
    ],
)
def test_local_forward_rotates_by_ardy_root_heading(
    heading: float, expected_endpoint: list[float]
) -> None:
    motion_rep, received = _conditions()
    root_history = torch.tensor([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])

    build_waypoint_constraints(
        motion_rep,
        root_history,
        torch.tensor(heading),
        (1.0, 0.0),
        generated_frames=125,
        history_frames=4,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        received["data"]["root_2d"][0],
        torch.tensor([expected_endpoint]),
        atol=1e-6,
        rtol=1e-6,
    )
