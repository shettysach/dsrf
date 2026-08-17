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
    EndEffectorSelection,
    EndEffectorTarget,
    GroundingRequest,
    GroundingResult,
    MotionChunk,
    PipelineError,
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
    command = AgentCommand(
        3,
        "reach",
        "reach with the right hand",
        (),
        end_effectors=(EndEffectorTarget("right_hand", (0.4, -0.2, 0.8)),),
    )
    value, metadata = agent_command_to_arrow(command)
    assert agent_command_from_arrow(value, metadata) == command


def test_observation_arrow_round_trip() -> None:
    observation = VisualObservation(4, "stand", b"jpeg")
    value, metadata = observation_to_arrow(observation)
    restored = observation_from_arrow(value, metadata)
    assert restored.observation_id == observation.observation_id
    assert restored.completed_command == observation.completed_command
    assert restored.jpeg == observation.jpeg


def test_grounding_messages_arrow_round_trip() -> None:
    from shared.arrow import (
        grounding_request_from_arrow,
        grounding_request_to_arrow,
        grounding_result_from_arrow,
        grounding_result_to_arrow,
    )

    request = GroundingRequest(4, ((300, 700), (600, 500)))
    value, metadata = grounding_request_to_arrow(request)
    assert grounding_request_from_arrow(value, metadata) == request

    result = GroundingResult(4, ((1.25, -0.5), (0.5, 1.0)))
    value, metadata = grounding_result_to_arrow(result)
    restored = grounding_result_from_arrow(value, metadata)
    assert restored.observation_id == result.observation_id
    np.testing.assert_allclose(restored.target_xys, result.target_xys)

    request = GroundingRequest(
        5,
        ((300, 700),),
        (EndEffectorSelection("left_hand", (400, 300)),),
    )
    value, metadata = grounding_request_to_arrow(request)
    assert grounding_request_from_arrow(value, metadata) == request

    result = GroundingResult(
        5,
        ((1.0, 0.0),),
        (EndEffectorTarget("left_hand", (0.5, 0.2, 0.7)),),
    )
    value, metadata = grounding_result_to_arrow(result)
    restored = grounding_result_from_arrow(value, metadata)
    assert restored.end_effectors[0].name == "left_hand"
    np.testing.assert_allclose(
        restored.end_effectors[0].target_xyz,
        result.end_effectors[0].target_xyz,
    )


def test_pipeline_error_arrow_round_trip() -> None:
    error = PipelineError("motion-gen", 2, "bad command")
    assert pipeline_error_from_arrow(pipeline_error_to_arrow(error)) == error
