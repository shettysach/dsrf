from pathlib import Path

import yaml


def test_runtime_environment_is_scoped_to_consuming_nodes() -> None:
    descriptor = yaml.safe_load(Path("demo.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert nodes["agent"]["env"] == {
        "VLM_URL": "http://127.0.0.1:8080",
        "VLM_TIMEOUT": "120",
        "VLM_SYSTEM_PROMPT": "TASK.md",
        "VLM_USER_PROMPT": "prompt/PLANNER_USER.md",
        "MOTION_GENERATOR": "kinematic_planner",
    }
    assert nodes["motion-gen"]["env"] == {
        "DEVICE": "cuda",
        "MOTION_GENERATOR": ("${MOTION_GENERATOR:-kinematic_planner}"),
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
        "REFERENCE_GHOST": "${REFERENCE_GHOST:-false}",
    }


def test_ardy_dataflow_wires_encoder_between_agent_and_motion_gen() -> None:
    descriptor = yaml.safe_load(Path("ardy.yml").read_text())
    nodes = {node["id"]: node for node in descriptor["nodes"]}

    assert set(nodes) == {"agent", "text-encoder", "motion-gen", "sim"}
    assert nodes["text-encoder"]["inputs"] == {"command": "agent/command"}
    assert nodes["text-encoder"]["outputs"] == ["encoded_command", "error"]
    assert nodes["text-encoder"]["env"]["TEXT_ENCODER_MODEL"] == (
        "${TEXT_ENCODER_MODEL:-/tmp/model}"
    )
    assert nodes["agent"]["env"]["VLM_USER_PROMPT"] == "prompt/USER.md"
    assert nodes["motion-gen"]["inputs"] == {
        "encoded_command": "text-encoder/encoded_command"
    }
    assert nodes["motion-gen"]["env"]["MOTION_GENERATOR"] == "ardy"
    assert nodes["sim"]["inputs"] == {"motion": "motion-gen/motion"}
    assert nodes["agent"]["inputs"]["observation"] == "sim/observation"
    assert nodes["agent"]["inputs"]["encoding_error"] == "text-encoder/error"
