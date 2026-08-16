from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest
import torch

import nodes.motion_gen as motion_gen_node
import sim.runtime as sim_runtime
from shared.arrow import (
    agent_command_to_arrow,
    grounding_request_to_arrow,
    grounding_result_from_arrow,
    motion_from_arrow,
    motion_to_arrow,
    observation_from_arrow,
    pipeline_error_from_arrow,
)
from shared.config import KinematicPlannerConfig
from shared.messages import (
    AgentCommand,
    EndEffectorSelection,
    EndEffectorTarget,
    GroundingRequest,
    MotionChunk,
    ProjectionContext,
)


class _Node:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = iter(events)
        self.outputs: list[tuple[str, object, dict[str, object]]] = []
        self.logs: list[tuple[str, str, dict[str, object]]] = []

    def __iter__(self):
        return self.events

    def send_output(self, output_id, value, **kwargs) -> None:
        self.outputs.append((output_id, value, kwargs))

    def log(self, level, message, **kwargs) -> None:
        self.logs.append((level, message, kwargs))


def _command_event(
    observation_id: int,
    text: str,
    motion: str = "walk",
    target_xys: tuple[tuple[float, float], ...] = ((1.0, 0.0),),
) -> dict[str, object]:
    value, metadata = agent_command_to_arrow(
        AgentCommand(observation_id, text, motion, target_xys)
    )
    return {
        "type": "INPUT",
        "id": "command",
        "value": value,
        "metadata": metadata,
    }


def _run_motion_gen(monkeypatch, events, generate):
    node = _Node(events)
    generator = SimpleNamespace(generate=generate, fps=30)
    config = motion_gen_node.MotionGenConfig(
        device="cpu",
        backend=KinematicPlannerConfig(planner_onnx=Path("planner.onnx")),
    )
    monkeypatch.setattr(motion_gen_node.MotionGenConfig, "from_env", lambda: config)
    monkeypatch.setattr(motion_gen_node, "Node", lambda: node)
    monkeypatch.setattr(
        motion_gen_node, "KinematicPlanner", lambda *args, **kwargs: generator
    )
    monkeypatch.setattr(motion_gen_node, "_create_text_encoder", lambda cfg: None)
    motion_gen_node.main()
    return node


def _planner_motion() -> np.ndarray:
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    return qpos


def test_motion_gen_generates_one_segment_per_command(monkeypatch) -> None:
    generated: list[tuple[str, tuple[tuple[float, float], ...], str | None]] = []

    def generate(motion, target_xys, direction):
        generated.append((motion, target_xys, direction))
        return _planner_motion()

    node = _run_motion_gen(
        monkeypatch,
        [_command_event(4, '{"motion":"walk","waypoints_2d":[[500,500]]}')],
        generate,
    )

    motions = [output for output in node.outputs if output[0] == "motion"]
    assert generated == [("walk", ((1.0, 0.0),), None)]
    assert len(motions) == 1
    _, value, kwargs = motions[0]
    chunk = motion_from_arrow(value, kwargs["metadata"])
    assert chunk.observation_id == 4
    assert chunk.command == '{"motion":"walk","waypoints_2d":[[500,500]]}'
    assert any("motion generated" in message for _, message, _ in node.logs)


def test_motion_gen_reports_invalid_raw_vlm_response(monkeypatch) -> None:
    def generate(motion, target_xys, direction):
        del motion, target_xys, direction
        raise ValueError("Command must be a JSON object")

    node = _run_motion_gen(
        monkeypatch,
        [_command_event(5, "I think the robot should walk")],
        generate,
    )

    errors = [output for output in node.outputs if output[0] == "error"]
    assert len(errors) == 1
    error = pipeline_error_from_arrow(errors[0][1])
    assert error.source == "motion-gen"
    assert error.observation_id == 5
    assert "Command must be a JSON object" in error.detail


def test_motion_gen_does_not_swallow_planner_errors(monkeypatch) -> None:
    def generate(motion, target_xys, direction):
        del motion, target_xys, direction
        raise KeyError("unexpected")

    with pytest.raises(KeyError, match="unexpected"):
        _run_motion_gen(
            monkeypatch,
            [_command_event(0, '{"motion":"walk","waypoints_2d":[[500,500]]}')],
            generate,
        )


def test_ardy_motion_gen_encodes_commands_in_process(monkeypatch) -> None:
    from shared.config import ArdyConfig

    command_text = '{"motion":"walk","waypoints_2d":[[700,500],[500,700]]}'
    node = _Node(
        [_command_event(7, command_text, target_xys=((0.2, -0.7), (0.6, 0.1)))]
    )
    embedding = torch.ones(4096)
    encoded: list[str] = []
    generated: list[
        tuple[
            torch.Tensor,
            tuple[tuple[float, float], ...],
            tuple[EndEffectorTarget, ...],
        ]
    ] = []
    generator = SimpleNamespace(
        fps=25,
        generate=lambda embedding, target, end_effectors: (
            generated.append((embedding, target, end_effectors)) or _planner_motion()
        ),
    )
    encoder = SimpleNamespace(
        encode=lambda text: encoded.append(text) or embedding,
    )
    config = motion_gen_node.MotionGenConfig(
        device="cpu",
        backend=ArdyConfig(
            checkpoints_dir=Path("checkpoints"),
            text_encoder_model=Path("text-encoder"),
            text_encoder_device="cuda:1",
        ),
    )
    monkeypatch.setattr(motion_gen_node.MotionGenConfig, "from_env", lambda: config)
    monkeypatch.setattr(motion_gen_node, "Node", lambda: node)
    monkeypatch.setattr(motion_gen_node, "_create_generator", lambda cfg: generator)
    monkeypatch.setattr(motion_gen_node, "_create_text_encoder", lambda cfg: encoder)

    motion_gen_node.main()

    assert encoded == ["walk"]
    assert len(generated) == 1
    assert generated[0][0] is embedding
    assert generated[0][1] == ((0.2, -0.7), (0.6, 0.1))
    assert generated[0][2] == ()
    chunk = motion_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert chunk.observation_id == 7
    assert chunk.command == command_text


class _Simulation:
    device = "cpu"

    def __init__(self) -> None:
        self.steps = 0

    def compute_context(self):
        return nullcontext()

    def robot_state(self):
        return SimpleNamespace(
            root_pos_w=torch.zeros(3),
            root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )

    def step(self, action) -> None:
        del action
        self.steps += 1


class _Policy:
    def __init__(self) -> None:
        self.calls = 0
        self.loaded: MotionChunk | None = None

    def load_motion(self, chunk, root_pos_w, root_quat_w) -> None:
        del root_pos_w, root_quat_w
        self.loaded = chunk

    def infer(self, state):
        del state
        self.calls += 1
        return torch.zeros((1, 29)), self.calls == 2


class _Renderer:
    def __init__(self, simulation: _Simulation) -> None:
        self.simulation = simulation
        self.capture_steps: list[int] = []
        self.jpeg_steps: list[int] = []
        self.rgbd_steps: list[int] = []
        self.depth_steps: list[int] = []

    def capture_jpeg(self) -> bytes:
        self.capture_steps.append(self.simulation.steps)
        self.jpeg_steps.append(self.simulation.steps)
        return f"jpeg-{self.simulation.steps}".encode()

    def capture_rgbd(self) -> tuple[bytes, ProjectionContext]:
        self.capture_steps.append(self.simulation.steps)
        self.rgbd_steps.append(self.simulation.steps)
        return f"jpeg-{self.simulation.steps}".encode(), _projection()

    def capture_depth(self) -> ProjectionContext:
        self.depth_steps.append(self.simulation.steps)
        return _projection()


def _projection() -> ProjectionContext:
    return ProjectionContext(
        depth=np.ones((101, 101), dtype=np.float32),
        camera_pos_w=np.zeros(3),
        camera_forward_w=np.array([1.0, 0.0, 0.0]),
        camera_up_w=np.array([0.0, 0.0, 1.0]),
        frustum_height=1.0,
        root_pos_w=np.zeros(3),
        root_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        near=0.01,
        far=100.0,
    )


class _Viewer:
    def __init__(self, simulation: _Simulation) -> None:
        self.simulation = simulation
        self.sync_steps: list[int] = []

    def sync(self) -> None:
        self.sync_steps.append(self.simulation.steps)

    def close(self) -> None:
        pass


def _motion_event(chunk: MotionChunk) -> dict[str, object]:
    value, metadata = motion_to_arrow(chunk)
    return {
        "type": "INPUT",
        "id": "motion",
        "value": value,
        "metadata": metadata,
    }


def _grounding_request_event(
    observation_id: int,
    waypoints_2d: tuple[tuple[int, int], ...] = ((500, 500),),
    end_effectors_2d: tuple[EndEffectorSelection, ...] = (),
) -> dict[str, object]:
    value, metadata = grounding_request_to_arrow(
        GroundingRequest(observation_id, waypoints_2d, end_effectors_2d)
    )
    return {
        "type": "INPUT",
        "id": "grounding_request",
        "value": value,
        "metadata": metadata,
    }


def test_sonic_steps_final_action_before_capture(monkeypatch) -> None:
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    chunk = MotionChunk(0, "walk forward", qpos)
    node = _Node([_motion_event(chunk), {"type": "STOP"}])
    simulation = _Simulation()
    policy = _Policy()
    renderer = _Renderer(simulation)
    viewer = _Viewer(simulation)
    monkeypatch.setattr(sim_runtime.time, "sleep", lambda delay: None)

    runtime = sim_runtime.SimRuntime(
        cast(Any, node),
        cast(Any, simulation),
        cast(Any, policy),
        cast(Any, renderer),
        cast(Any, viewer),
    )
    runtime.run()

    assert simulation.steps == 2
    assert viewer.sync_steps == [1, 2]
    assert renderer.jpeg_steps == [0, 2]
    assert renderer.depth_steps == []
    observations = [output for output in node.outputs if output[0] == "observation"]
    first = observation_from_arrow(
        observations[0][1], cast(Any, observations[0][2]["metadata"])
    )
    second = observation_from_arrow(
        observations[1][1], cast(Any, observations[1][2]["metadata"])
    )
    assert first.observation_id == 0
    assert first.completed_command is None
    assert second.observation_id == 1
    assert second.completed_command == "walk forward"
    assert any("[OBS 0->1] motion complete" in message for _, message, _ in node.logs)


def test_sim_publishes_rgb_without_eager_depth() -> None:
    node = _Node([{"type": "STOP"}])
    simulation = _Simulation()
    renderer = _Renderer(simulation)
    runtime = sim_runtime.SimRuntime(
        cast(Any, node),
        cast(Any, simulation),
        cast(Any, _Policy()),
        cast(Any, renderer),
    )

    runtime.run()

    assert renderer.jpeg_steps == [0]
    assert renderer.rgbd_steps == []
    observation = observation_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert observation.jpeg == b"jpeg-0"


def test_sim_lazily_caches_depth_for_current_observation() -> None:
    node = _Node(
        [
            _grounding_request_event(0),
            _grounding_request_event(0, ((600, 500), (400, 500))),
            {"type": "STOP"},
        ]
    )
    simulation = _Simulation()
    renderer = _Renderer(simulation)
    runtime = sim_runtime.SimRuntime(
        cast(Any, node),
        cast(Any, simulation),
        cast(Any, _Policy()),
        cast(Any, renderer),
    )

    runtime.run()

    assert simulation.steps == 0
    assert renderer.jpeg_steps == [0]
    assert renderer.depth_steps == [0]
    results = [output for output in node.outputs if output[0] == "grounding_result"]
    assert len(results) == 2
    assert all(
        grounding_result_from_arrow(value, cast(Any, kwargs["metadata"])).observation_id
        == 0
        for _, value, kwargs in results
    )


def test_sim_grounds_end_effector_with_lazy_depth() -> None:
    node = _Node(
        [
            _grounding_request_event(
                0,
                (),
                (EndEffectorSelection("right_hand", (500, 500)),),
            ),
            {"type": "STOP"},
        ]
    )
    simulation = _Simulation()
    renderer = _Renderer(simulation)
    runtime = sim_runtime.SimRuntime(
        cast(Any, node),
        cast(Any, simulation),
        cast(Any, _Policy()),
        cast(Any, renderer),
    )

    runtime.run()

    assert renderer.depth_steps == [0]
    output = next(output for output in node.outputs if output[0] == "grounding_result")
    result = grounding_result_from_arrow(
        output[1], cast(Any, output[2]["metadata"])
    )
    assert result.end_effectors[0].name == "right_hand"
    np.testing.assert_allclose(
        result.end_effectors[0].target_xyz, (1.0, 0.0, 0.0), atol=0.3
    )


def test_sonic_rejects_motion_for_stale_observation() -> None:
    node = _Node([])
    simulation = _Simulation()
    runtime = sim_runtime.SimRuntime(
        cast(Any, node),
        cast(Any, simulation),
        cast(Any, _Policy()),
        cast(Any, _Renderer(simulation)),
    )
    runtime._accept_motion(
        {
            "type": "INPUT",
            "id": "motion",
            "value": pa.array(np.zeros(72), type=pa.float32()),
            "metadata": {"observation_id": "3", "command": "stand"},
        }
    )

    error = pipeline_error_from_arrow(node.outputs[-1][1])
    assert error.observation_id == 0
    assert "Expected motion for observation 0, got 3" in error.detail
