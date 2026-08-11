import numpy as np

from shared.arrow import (
    agent_command_from_arrow,
    agent_command_to_arrow,
    motion_from_arrow,
    motion_to_arrow,
    observation_from_arrow,
    observation_to_arrow,
    pipeline_error_from_arrow,
    pipeline_error_to_arrow,
)
from shared.messages import (
    AgentCommand,
    MotionChunk,
    PipelineError,
    ProjectionContext,
    VisualObservation,
)


def test_motion_arrow_round_trip() -> None:
    chunk = MotionChunk(
        7,
        "walk forward 0.4",
        np.arange(72, dtype=np.float32).reshape(2, 36),
    )
    value, metadata = motion_to_arrow(chunk)
    restored = motion_from_arrow(value, metadata)
    assert restored.observation_id == 7
    assert restored.command == "walk forward 0.4"
    np.testing.assert_array_equal(restored.qpos, chunk.qpos)


def test_agent_command_arrow_round_trip() -> None:
    command = AgentCommand(3, "waypoint", "walk", (0.4, 0.2))
    value, metadata = agent_command_to_arrow(command)
    assert agent_command_from_arrow(value, metadata) == command


def test_observation_arrow_round_trip() -> None:
    projection = ProjectionContext(
        depth=np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0,
        camera_pos_w=np.array([1.0, 2.0, 3.0]),
        camera_forward_w=np.array([1.0, 0.0, 0.0]),
        camera_up_w=np.array([0.0, 0.0, 1.0]),
        frustum_height=1.0,
        root_pos_w=np.zeros(3),
        root_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        near=0.01,
        far=100.0,
    )
    observation = VisualObservation(4, "stand", b"jpeg", projection)
    value, metadata = observation_to_arrow(observation)
    restored = observation_from_arrow(value, metadata)
    assert restored.observation_id == observation.observation_id
    assert restored.completed_command == observation.completed_command
    assert restored.jpeg == observation.jpeg
    assert restored.projection is not None
    np.testing.assert_array_equal(restored.projection.depth, projection.depth)
    np.testing.assert_allclose(restored.projection.camera_pos_w, projection.camera_pos_w)


def test_pipeline_error_arrow_round_trip() -> None:
    error = PipelineError("motion-gen", 2, "bad command")
    assert pipeline_error_from_arrow(pipeline_error_to_arrow(error)) == error
