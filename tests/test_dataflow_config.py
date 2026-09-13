import tomllib
from pathlib import Path

import yaml


def test_only_push_motion_script_launcher_remains() -> None:
    assert [path.name for path in Path(".").glob("*.yml")] == ["push_motion_script.yml"]
    nodes = {
        node["id"]: node
        for node in yaml.safe_load(Path("push_motion_script.yml").read_text())["nodes"]
    }
    assert set(nodes) == {"agent", "sim"}
    assert nodes["agent"]["env"]["SCRIPT_TASK"] == "push_motion"
    assert nodes["agent"]["env"]["SCRIPT_START_IMMEDIATELY"] == "true"
    assert nodes["sim"]["env"]["TASK"] == "push_motion"
    assert nodes["sim"]["env"]["PUBLISH_OBSERVATIONS"] == "false"
    assert nodes["sim"]["env"]["PUSH_MOTION_HANDS"] == "${PUSH_MOTION_HANDS:-true}"
    assert nodes["sim"]["outputs"] == ["error"]


def test_text_encoder_is_a_library_without_a_node_entry_point() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert "dsrf-text-encoder" not in project["project"]["scripts"]
    packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/encoder" not in packages
    assert all(Path(package).is_dir() for package in packages)
