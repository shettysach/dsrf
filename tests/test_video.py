import numpy as np

from sim.video import DemoVlmState, compose_demo_frame

LONG_REASONING = " ".join(["Carefully inspect alignment before moving forward."] * 40)


def test_demo_overlay_stays_compact_at_640_by_480() -> None:
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)

    composed = compose_demo_frame(
        frame,
        DemoVlmState(12, LONG_REASONING, '{"motion":"turn","direction":"left"}'),
    )

    changed = np.any(composed != frame, axis=2)
    ys, xs = np.where(changed)
    assert composed.shape == frame.shape
    assert xs.max() - xs.min() + 1 <= round(640 * 0.28)
    assert round(480 * 0.32) < ys.max() - ys.min() + 1 <= round(480 * 0.55)
    assert ys.min() >= 0


def test_demo_overlay_scales_up_without_dominating_720p() -> None:
    frame = np.full((720, 1280, 3), 220, dtype=np.uint8)

    composed = compose_demo_frame(
        frame,
        DemoVlmState(12, LONG_REASONING, '{"motion":"walk","direction":"forward"}'),
    )

    changed = np.any(composed != frame, axis=2)
    ys, xs = np.where(changed)
    assert composed.shape == frame.shape
    assert xs.max() - xs.min() + 1 <= round(1280 * 0.28)
    assert ys.max() - ys.min() + 1 <= round(720 * 0.55)
