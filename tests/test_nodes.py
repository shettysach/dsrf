from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import sim.runtime as sim_runtime
from motion_gen.ardy.adapter import ArdyMotionGenerator
from motion_gen.kinematic_planner.adapter import KinematicPlannerMotionGenerator
from shared.arrow import (
    agent_command_to_arrow,
    grounding_request_to_arrow,
    grounding_result_from_arrow,
    observation_from_arrow,
    pipeline_error_from_arrow,
)
from shared.messages import (
    AgentCommand,
    EndEffectorSelection,
    EndEffectorTarget,
    GroundingRequest,
)
from sim.camera import ProjectionContext


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


def _command_event(observation_id: int, text: str) -> dict[str, object]:
    value, metadata = agent_command_to_arrow(
        AgentCommand(observation_id, text, "walk", ((1.0, 0.0),))
    )
    return {
        "type": "INPUT",
        "id": "command",
        "value": value,
        "metadata": metadata,
    }


def _planner_motion() -> torch.Tensor:
    qpos = torch.zeros((2, 36))
    qpos[:, 3] = 1.0
    return qpos


def test_motion_generator_adapters_validate_backend_specific_constraints() -> None:
    planner = KinematicPlannerMotionGenerator(
        SimpleNamespace(fps=30, generate=lambda *args: _planner_motion())
    )
    ardy = ArdyMotionGenerator(
        SimpleNamespace(fps=25, generate=lambda *args: _planner_motion()),
        SimpleNamespace(encode=lambda text: torch.ones(4096)),
    )

    with pytest.raises(ValueError, match="End-effector constraints"):
        planner.generate(
            AgentCommand(
                0,
                "wave",
                "wave",
                (),
                end_effectors=(EndEffectorTarget("left_hand", (0.1, 0.0, 0.5)),),
            )
        )
    with pytest.raises(ValueError, match="Directional commands"):
        ardy.generate(AgentCommand(0, "walk left", "walk", (), direction="left"))


class _Simulation:
    device = "cpu"
    step_dt = 0.02

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


class _Generator:
    fps = 50
    last_encode_ms: float | None = None

    def __init__(self, qpos: torch.Tensor | None = None) -> None:
        self.qpos = _planner_motion() if qpos is None else qpos
        self.commands: list[AgentCommand] = []

    def generate(self, command: AgentCommand) -> torch.Tensor:
        self.commands.append(command)
        return self.qpos


class _Tracker:
    def __init__(self) -> None:
        self.calls = 0
        self.loaded: torch.Tensor | None = None

    def load_motion(self, qpos: torch.Tensor, state) -> None:
        del state
        self.loaded = qpos

    def act(self, state):
        del state
        self.calls += 1
        return torch.zeros(29), self.calls == 2


class _Renderer:
    def __init__(self, simulation: _Simulation) -> None:
        self.simulation = simulation
        self.rgbd_steps: list[int] = []

    def capture_rgbd(self) -> tuple[bytes, ProjectionContext]:
        self.rgbd_steps.append(self.simulation.steps)
        return f"jpeg-{self.simulation.steps}".encode(), _projection()


def _projection() -> ProjectionContext:
    return ProjectionContext(
        depth=torch.ones((101, 101)),
        camera_pos_w=torch.zeros(3),
        camera_rotation_w=torch.tensor(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        fovy_rad=2.0 * np.arctan(0.5),
        root_pos_w=torch.zeros(3),
        root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
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


def _runtime(node, simulation, generator, tracker, renderer, viewer=None):
    return sim_runtime.SimRuntime(
        cast(Any, node),
        cast(Any, simulation),
        cast(Any, generator),
        cast(Any, tracker),
        cast(Any, renderer),
        cast(Any, viewer) if viewer is not None else None,
    )


def test_sim_generates_tracks_and_steps_final_action(monkeypatch) -> None:
    node = _Node([_command_event(0, "walk forward"), {"type": "STOP"}])
    simulation = _Simulation()
    generator = _Generator()
    tracker = _Tracker()
    renderer = _Renderer(simulation)
    viewer = _Viewer(simulation)
    monkeypatch.setattr(sim_runtime.time, "sleep", lambda delay: None)

    _runtime(node, simulation, generator, tracker, renderer, viewer).run()

    assert generator.commands[0].text == "walk forward"
    assert tracker.loaded is not None
    torch.testing.assert_close(tracker.loaded, generator.qpos)
    assert simulation.steps == 2
    assert viewer.sync_steps == [1, 2]
    assert renderer.rgbd_steps == [0, 2]
    observations = [output for output in node.outputs if output[0] == "observation"]
    first = observation_from_arrow(
        observations[0][1], cast(Any, observations[0][2]["metadata"])
    )
    second = observation_from_arrow(
        observations[1][1], cast(Any, observations[1][2]["metadata"])
    )
    assert first.completed_command is None
    assert second.completed_command == "walk forward"


def test_sim_reports_generator_value_errors() -> None:
    class _FailingGenerator(_Generator):
        def generate(self, command: AgentCommand) -> torch.Tensor:
            raise ValueError("unsupported motion")

    node = _Node([_command_event(0, "wave"), {"type": "STOP"}])
    simulation = _Simulation()
    _runtime(
        node,
        simulation,
        _FailingGenerator(),
        _Tracker(),
        _Renderer(simulation),
    ).run()

    error = pipeline_error_from_arrow(node.outputs[-1][1])
    assert error.source == "motion-gen"
    assert error.observation_id == 0
    assert error.detail == "unsupported motion"


def test_sim_publishes_synchronized_rgbd() -> None:
    node = _Node([{"type": "STOP"}])
    simulation = _Simulation()
    renderer = _Renderer(simulation)
    _runtime(node, simulation, _Generator(), _Tracker(), renderer).run()

    assert renderer.rgbd_steps == [0]
    observation = observation_from_arrow(
        node.outputs[0][1], cast(Any, node.outputs[0][2]["metadata"])
    )
    assert observation.jpeg == b"jpeg-0"


def test_sim_reuses_depth_for_current_observation() -> None:
    node = _Node(
        [
            _grounding_request_event(0),
            _grounding_request_event(0, ((600, 500), (400, 500))),
            {"type": "STOP"},
        ]
    )
    simulation = _Simulation()
    renderer = _Renderer(simulation)
    _runtime(node, simulation, _Generator(), _Tracker(), renderer).run()

    assert renderer.rgbd_steps == [0]
    results = [output for output in node.outputs if output[0] == "grounding_result"]
    assert len(results) == 2
    assert all(
        grounding_result_from_arrow(value, cast(Any, kwargs["metadata"])).observation_id
        == 0
        for _, value, kwargs in results
    )


def test_sim_grounds_end_effector_with_published_depth() -> None:
    node = _Node(
        [
            _grounding_request_event(
                0, (), (EndEffectorSelection("right_hand", (500, 500)),)
            ),
            {"type": "STOP"},
        ]
    )
    simulation = _Simulation()
    output = _runtime(
        node, simulation, _Generator(), _Tracker(), _Renderer(simulation)
    )
    output.run()

    result_output = next(output for output in node.outputs if output[0] == "grounding_result")
    result = grounding_result_from_arrow(
        result_output[1], cast(Any, result_output[2]["metadata"])
    )
    assert result.end_effectors[0].name == "right_hand"
    np.testing.assert_allclose(
        result.end_effectors[0].target_xyz, (1.0, 0.0, 0.0), atol=0.3
    )


def test_sim_rejects_command_for_stale_observation() -> None:
    node = _Node([])
    simulation = _Simulation()
    runtime = _runtime(
        node, simulation, _Generator(), _Tracker(), _Renderer(simulation)
    )
    runtime._accept_command(_command_event(3, "stand"))

    error = pipeline_error_from_arrow(node.outputs[-1][1])
    assert error.observation_id == 0
    assert "Expected command for observation 0, got 3" in error.detail
