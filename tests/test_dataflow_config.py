import tomllib
from pathlib import Path

import yaml


def test_runtime_environment_is_scoped_to_consuming_nodes() -> None:
    descriptor = yaml.safe_load(Path("corridors.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"] == {
        "VLM_URL": "http://127.0.0.1:8080",
        "VLM_TIMEOUT": "120",
        "AGENT_DEBUG": "${AGENT_DEBUG:-false}",
        "VLM_HISTORY_TURNS": "${VLM_HISTORY_TURNS:-8}",
        "VLM_HISTORY_RETAIN_TURNS": "${VLM_HISTORY_RETAIN_TURNS:-2}",
        "VLM_SYSTEM_PROMPT": "tasks/portrait_corridors/TASK.md",
        "VLM_USER_PROMPT": "prompt/PLANNER_USER.md",
        "MOTION_GENERATOR": "kinematic_planner",
    }
    assert nodes["motion-gen"]["env"] == {
        "DEVICE": "cuda",
        "MOTION_GENERATOR": "kinematic_planner",
        "PLANNER_ONNX": ("/tmp/GEAR-SONIC/planner_sonic.onnx"),
    }
    assert nodes["sim"]["env"] == {
        "DEVICE": "cuda",
        "SONIC_DIR": "/tmp/GEAR-SONIC",
        "TASK": "portrait-corridors",
        "GOAL_INDEX": "${GOAL_INDEX:-1}",
        "IMAGE_WIDTH": "1280",
        "IMAGE_HEIGHT": "720",
        "JPEG_QUALITY": "85",
        "VIEWER": "native",
        "REFERENCE_GHOST": "${REFERENCE_GHOST:-false}",
        "CAMERA_YAW": "${CAMERA_YAW:-true}",
        "DEMO_VIDEO_DIR": "${DEMO_VIDEO_DIR:-}",
        "DEMO_RUNS": "${DEMO_RUNS:-10}",
        "MOTION_TIMEOUT_SECONDS": "${MOTION_TIMEOUT_SECONDS:-20}",
    }


def test_ardy_dataflow_encodes_commands_in_motion_gen() -> None:
    descriptor = yaml.safe_load(Path("ardy.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "motion-gen", "sim"}
    assert nodes["agent"]["env"]["VLM_USER_PROMPT"] == "prompt/USER.md"
    assert (
        nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/portrait_corridors/TASK.md"
    )
    assert nodes["motion-gen"]["inputs"] == {"command": "agent/command"}
    assert nodes["motion-gen"]["env"]["MOTION_GENERATOR"] == "ardy"
    assert nodes["motion-gen"]["env"]["TEXT_ENCODER_MODEL"] == "/tmp/model"
    assert nodes["motion-gen"]["env"]["TEXT_ENCODER_DEVICE"] == "cuda"
    assert nodes["sim"]["inputs"] == {
        "motion": "motion-gen/motion",
        "grounding_request": "agent/grounding_request",
    }
    assert nodes["agent"]["inputs"]["observation"] == "sim/observation"
    assert nodes["agent"]["inputs"]["grounding_result"] == "sim/grounding_result"
    assert "encoding_error" not in nodes["agent"]["inputs"]


def test_sokoban_dataflow_uses_kinematic_planner_without_scouting() -> None:
    descriptor = yaml.safe_load(Path("sokoban.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "motion-gen", "sim"}
    assert nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/sokoban/TASK.md"
    assert nodes["agent"]["env"]["AGENT_DEBUG"] == "${AGENT_DEBUG:-false}"
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "kinematic_planner"
    assert nodes["sim"]["env"]["TASK"] == "sokoban"
    assert nodes["sim"]["env"]["GOAL_INDEX"] == "${GOAL_INDEX:-0}"


def test_stack_steps_dataflow_uses_stack_steps_task() -> None:
    descriptor = yaml.safe_load(Path("stack_steps.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "motion-gen", "sim"}
    assert nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == ("tasks/stack_steps/TASK.md")
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "kinematic_planner"
    assert nodes["sim"]["env"]["TASK"] == "stack-steps"


def test_see_saw_dataflow_uses_see_saw_task() -> None:
    descriptor = yaml.safe_load(Path("see_saw.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "motion-gen", "sim"}
    assert nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/see_saw/TASK.md"
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "kinematic_planner"
    assert nodes["sim"]["env"]["TASK"] == "see-saw"


def test_seesaw_video_dataflow_has_finite_recording_duration() -> None:
    descriptor = yaml.safe_load(Path("seesaw.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"capture"}
    assert nodes["capture"]["args"] == "run python scripts/record_seesaw_gif.py"
    assert nodes["capture"]["env"]["SEESAW_GIF_PATH"] == (
        "${SEESAW_GIF_PATH:-/tmp/see-saw.gif}"
    )
    assert nodes["capture"]["env"]["SEESAW_GIF_SECONDS"] == ("${SEESAW_GIF_SECONDS:-3}")


def test_sokoban_2d_dataflow_enables_waypoint_projection() -> None:
    descriptor = yaml.safe_load(Path("sokoban_2d.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "motion-gen", "sim"}
    assert nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/sokoban/SYSTEM_2D.md"
    assert nodes["agent"]["env"]["VLM_USER_PROMPT"] == "prompt/USER_2D.md"
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "kinematic_planner"
    assert nodes["sim"]["env"]["TASK"] == "sokoban"
    assert nodes["sim"]["env"]["CAPTURE_DEPTH"] == "true"
    assert nodes["sim"]["env"]["GOAL_INDEX"] == "${GOAL_INDEX:-0}"


def test_dataflow_system_prompts_exist() -> None:
    for path in (
        "tasks/portrait_corridors/TASK.md",
        "tasks/sokoban/TASK.md",
        "tasks/sokoban/SYSTEM_2D.md",
        "tasks/grid_sokoban/SYSTEM.md",
        "tasks/stack_steps/TASK.md",
        "tasks/see_saw/TASK.md",
    ):
        assert Path(path).is_file()


def test_text_encoder_is_a_library_without_a_node_entry_point() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())

    assert "dsrf-text-encoder" not in project["project"]["scripts"]
    packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/encoder" not in packages
    assert all(Path(package).is_dir() for package in packages)
