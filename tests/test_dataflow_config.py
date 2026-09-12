import tomllib
from pathlib import Path

import yaml
from tasks import get_task


def test_runtime_environment_is_scoped_to_consuming_nodes() -> None:
    descriptor = yaml.safe_load(Path("demo.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"] == {
        "VLM_URL": "http://127.0.0.1:8080",
        "VLM_TIMEOUT": "120",
        "VLM_SYSTEM_PROMPT": "tasks/portrait_corridors/TASK.md",
        "VLM_USER_PROMPT": "prompt/PLANNER_USER.md",
        "MOTION_GENERATOR": "kinematic_planner",
    }
    assert nodes["sim"]["env"] == {
        "DEVICE": "cuda",
        "MOTION_GENERATOR": "kinematic_planner",
        "PLANNER_ONNX": "/tmp/GEAR-SONIC/planner_sonic.onnx",
        "SONIC_DIR": "/tmp/GEAR-SONIC",
        "TASK": "portrait-corridors",
        "IMAGE_WIDTH": "640",
        "IMAGE_HEIGHT": "480",
        "JPEG_QUALITY": "85",
        "VIEWER": "native",
        "REFERENCE_GHOST": "${REFERENCE_GHOST:-false}",
    }


def test_demo_dataflow_configures_the_agent_system_prompt() -> None:
    descriptor = yaml.safe_load(Path("demo.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert (
        nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/portrait_corridors/TASK.md"
    )
    assert "TASK" not in nodes["agent"]["env"]


def test_ardy_dataflow_encodes_commands_in_sim() -> None:
    descriptor = yaml.safe_load(Path("ardy.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "sim"}
    assert nodes["agent"]["env"]["VLM_USER_PROMPT"] == "prompt/USER.md"
    assert (
        nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/portrait_corridors/TASK.md"
    )
    assert nodes["sim"]["inputs"] == {
        "command": "agent/command",
        "grounding_request": "agent/grounding_request",
    }
    assert nodes["sim"]["env"]["MOTION_GENERATOR"] == "ardy"
    assert nodes["sim"]["env"]["TEXT_ENCODER_MODEL"] == "/tmp/model"
    assert nodes["sim"]["env"]["TEXT_ENCODER_DEVICE"] == "cuda"
    assert nodes["agent"]["inputs"]["observation"] == "sim/observation"
    assert nodes["agent"]["inputs"]["grounding_result"] == "sim/grounding_result"
    assert "encoding_error" not in nodes["agent"]["inputs"]


def test_push_script_dataflow_runs_one_observation_driven_motion() -> None:
    descriptor = yaml.safe_load(Path("push_script.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"]["SCRIPT_START_IMMEDIATELY"] == "false"
    assert "BOX_PUSH_GOAL_X" not in nodes["agent"]["env"]
    assert "BOX_PUSH_GOAL_X" not in nodes["sim"]["env"]
    assert nodes["agent"]["inputs"]["observation"] == "sim/observation"
    assert nodes["sim"]["env"]["PUBLISH_OBSERVATIONS"] == "true"
    assert nodes["sim"]["env"]["DEMO_MAX_COMMANDS"] == "1"
    assert nodes["sim"]["env"]["PUSH_CONTACT_RETRIES"] == "0"
    assert nodes["sim"]["env"]["PUSH_REACQUISITIONS"] == "0"
    assert "observation" in nodes["sim"]["outputs"]


def test_push_motion_dataflow_uses_the_contact_free_task() -> None:
    descriptor = yaml.safe_load(Path("push_motion_script.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"]["SCRIPT_TASK"] == "push_motion"
    assert nodes["sim"]["env"]["TASK"] == "push_motion"
    assert "PUSH_WELD" not in nodes["sim"]["env"]


def test_push_diagnosis_uses_an_unconstrained_prompt() -> None:
    descriptor = yaml.safe_load(Path("push_diagnosis.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"]["SCRIPT_TASK"] == "prompt"
    assert nodes["sim"]["env"]["TASK"] == "push_motion"


def test_sokoban_dataflow_uses_kinematic_planner_without_scouting() -> None:
    descriptor = yaml.safe_load(Path("sokoban.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "sim"}
    assert nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/sokoban/TASK.md"
    assert nodes["agent"]["env"]["VLM_RECENT_TURNS"] == "${VLM_RECENT_TURNS:-3}"
    assert nodes["agent"]["env"]["VLM_ENABLE_FINISHED"] == "true"
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "kinematic_planner"
    assert nodes["agent"]["env"]["DEMO_MAX_COMMANDS"] == "${DEMO_MAX_COMMANDS:-}"
    assert nodes["sim"]["env"]["TASK"] == "sokoban"
    assert nodes["sim"]["env"]["SOKOBAN_LEVEL"] == "${SOKOBAN_LEVEL:-1}"
    assert nodes["sim"]["env"]["DEMO_VIDEO_PATH"] == "${DEMO_VIDEO_PATH:-}"
    assert nodes["sim"]["env"]["STOP_ON_STAND"] == "${STOP_ON_STAND:-false}"


def test_seesaw_dataflow_uses_ardy() -> None:
    descriptor = yaml.safe_load(Path("seesaw.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/seesaw/TASK.md"
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "ardy"
    assert nodes["sim"]["env"]["MOTION_GENERATOR"] == "ardy"
    assert nodes["sim"]["env"]["TASK"] == "seesaw"


def test_seesaw_observation_camera_tracks_with_a_fixed_azimuth() -> None:
    camera = get_task("seesaw").observation_camera

    assert camera.azimuth == 0.0


def test_stairs_dataflow_uses_ardy() -> None:
    descriptor = yaml.safe_load(Path("stairs.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/stairs/TASK.md"
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "ardy"
    assert nodes["sim"]["env"]["TASK"] == "stairs"


def test_dataflow_system_prompts_exist() -> None:
    for path in (
        "tasks/portrait_corridors/TASK.md",
        "tasks/sokoban/TASK.md",
        "tasks/seesaw/TASK.md",
        "tasks/stairs/TASK.md",
    ):
        assert Path(path).is_file()


def test_text_encoder_is_a_library_without_a_node_entry_point() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())

    assert "dsrf-text-encoder" not in project["project"]["scripts"]
    packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/encoder" not in packages
    assert all(Path(package).is_dir() for package in packages)
