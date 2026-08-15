import imageio.v3 as iio

from sim.trajectory import TrajectoryRenderer


def test_trajectory_renderer_returns_a_path_only_png() -> None:
    trajectory = TrajectoryRenderer(resolution=64)
    trajectory.append((0.0, 0.0))
    trajectory.append((1.0, 0.5))

    image = iio.imread(trajectory.render_png(), extension=".png")

    assert image.shape == (64, 64, 3)
    assert image.max() == 255
    assert image.min() == 0


def test_trajectory_renderer_reset_discards_the_previous_path() -> None:
    trajectory = TrajectoryRenderer(resolution=64)
    trajectory.append((0.0, 0.0))
    trajectory.append((10.0, 0.0))
    trajectory.reset()
    trajectory.append((0.0, 0.0))

    image = iio.imread(trajectory.render_png(), extension=".png")

    assert (image == 255).all(axis=2).sum() == 9
