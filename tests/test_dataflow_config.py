import tomllib
from pathlib import Path

import yaml

from tasks import get_task
from sim.config import make_sim_env_cfg


def test_runtime_environment_is_scoped_to_consuming_nodes() -> None:
    descriptor = yaml.safe_load(Path("corridors.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"] == {
        "VLM_URL": "http://127.0.0.1:8080",
        "VLM_TIMEOUT": "120",
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
        "IMAGE_WIDTH": "640",
        "IMAGE_HEIGHT": "480",
        "JPEG_QUALITY": "85",
        "VIEWER": "native",
        "REFERENCE_GHOST": "false",
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
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "kinematic_planner"
    assert nodes["sim"]["env"]["TASK"] == "sokoban"


def test_seesaw_dataflow_uses_ardy() -> None:
    descriptor = yaml.safe_load(Path("seesaw.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"]["VLM_SYSTEM_PROMPT"] == "tasks/seesaw/TASK.md"
    assert nodes["agent"]["env"]["MOTION_GENERATOR"] == "ardy"
    assert nodes["motion-gen"]["env"]["MOTION_GENERATOR"] == "ardy"
    assert nodes["sim"]["env"]["TASK"] == "seesaw"


def test_seesaw_tracks_position_with_a_fixed_azimuth() -> None:
    cfg = make_sim_env_cfg(task=get_task("seesaw"))

    assert cfg.viewer.origin_type is cfg.viewer.OriginType.ASSET_BODY
    assert cfg.viewer.entity_name == "robot"
    assert cfg.viewer.body_name == "torso_link"
    assert cfg.viewer.azimuth == 0.0


def test_dataflow_system_prompts_exist() -> None:
    for path in (
        "tasks/portrait_corridors/TASK.md",
        "tasks/sokoban/TASK.md",
        "tasks/seesaw/TASK.md",
    ):
        assert Path(path).is_file()


def test_text_encoder_is_a_library_without_a_node_entry_point() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())

    assert "dsrf-text-encoder" not in project["project"]["scripts"]
    packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/encoder" not in packages
    assert all(Path(package).is_dir() for package in packages)
