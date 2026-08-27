import numpy as np

from sim.video import DemoVideoRecorder, DemoVlmState, _format_ardy, compose_demo_frame


def test_video_frame_overlays_a_vlm_decision() -> None:
    source = np.zeros((120, 200, 3), dtype=np.uint8)
    rendered = compose_demo_frame(
        source,
        DemoVlmState(
            observation_id=4,
            reasoning="The target is on the robot's right.",
            command='{"motion":"walk","waypoints_2d":[[700,500]]}',
        ),
    )

    assert rendered.shape == source.shape
    assert rendered.dtype == np.uint8
    assert np.any(rendered != source)


def test_ardy_command_is_formatted_for_the_overlay() -> None:
    assert _format_ardy('{"motion":"wave","end_effectors":[]}') == "motion: wave"


def test_video_marks_waypoints_and_labeled_end_effectors() -> None:
    source = np.zeros((400, 600, 3), dtype=np.uint8)
    rendered = compose_demo_frame(
        source,
        DemoVlmState(
            command=(
                '{"motion":"reach","waypoints_2d":[[700,800]],'
                '"end_effectors":[{"name":"right_hand","target_2d":[800,700]}]}'
            )
        ),
    )

    assert tuple(rendered[320, 419]) == (35, 120, 255)
    assert tuple(rendered[279, 479]) == (235, 30, 30)


def test_video_can_hide_targets_after_the_decision_frame() -> None:
    source = np.zeros((400, 600, 3), dtype=np.uint8)
    rendered = compose_demo_frame(
        source,
        DemoVlmState(command='{"motion":"walk","waypoints_2d":[[700,800]]}'),
        show_targets=False,
    )

    assert tuple(rendered[320, 419]) != (35, 120, 255)


def test_video_shows_targets_for_half_a_second_per_observation(monkeypatch, tmp_path) -> None:
    frames: list[np.ndarray] = []

    class Writer:
        def append_data(self, frame: np.ndarray) -> None:
            frames.append(frame)

        def close(self) -> None:
            pass

    monkeypatch.setattr("sim.video.iio.get_writer", lambda *args, **kwargs: Writer())
    recorder = DemoVideoRecorder(tmp_path / "demo.mp4", fps=10)
    state = DemoVlmState(command='{"motion":"walk","waypoints_2d":[[700,800]]}')
    source = np.zeros((400, 600, 3), dtype=np.uint8)

    for _ in range(6):
        recorder.write_frame(source, state)

    assert tuple(frames[0][320, 419]) == (35, 120, 255)
    assert tuple(frames[4][320, 419]) == (35, 120, 255)
    assert tuple(frames[5][320, 419]) != (35, 120, 255)
