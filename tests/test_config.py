from pathlib import Path

import pytest
from tasks import get_task

from shared.config import (
    AgentConfig,
    ArdyConfig,
    KinematicPlannerConfig,
    MotionGenConfig,
    SimConfig,
)


def test_motion_gen_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEVICE", "cuda:0")
    monkeypatch.setenv("MOTION_GENERATOR", "kinematic_planner")
    monkeypatch.setenv("PLANNER_ONNX", "/models/planner.onnx")

    assert MotionGenConfig.from_env() == MotionGenConfig(
        device="cuda:0",
        backend=KinematicPlannerConfig(planner_onnx=Path("/models/planner.onnx")),
    )


def test_ardy_motion_gen_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("MOTION_GENERATOR", "ardy")
    monkeypatch.setenv("CHECKPOINTS_DIR", "/models/ardy")
    monkeypatch.setenv("TEXT_ENCODER_MODEL", "/models/text-encoder")
    monkeypatch.setenv("TEXT_ENCODER_DEVICE", "cuda:1")

    assert MotionGenConfig.from_env() == MotionGenConfig(
        device="cpu",
        backend=ArdyConfig(
            checkpoints_dir=Path("/models/ardy"),
            text_encoder_model=Path("/models/text-encoder"),
            text_encoder_device="cuda:1",
        ),
    )


def test_ardy_motion_gen_config_requires_no_fixed_conditioning(monkeypatch) -> None:
    monkeypatch.setenv("DEVICE", "cuda:0")
    monkeypatch.setenv("MOTION_GENERATOR", "ardy")
    monkeypatch.setenv("CHECKPOINTS_DIR", "/models/ardy")
    monkeypatch.setenv("TEXT_ENCODER_MODEL", "/models/text-encoder")
    monkeypatch.setenv("TEXT_ENCODER_DEVICE", "cpu")
    assert MotionGenConfig.from_env().backend == ArdyConfig(
        checkpoints_dir=Path("/models/ardy"),
        text_encoder_model=Path("/models/text-encoder"),
        text_encoder_device="cpu",
    )


def test_motion_gen_config_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("MOTION_GENERATOR", "unknown")

    with pytest.raises(ValueError, match="MOTION_GENERATOR"):
        MotionGenConfig.from_env()


def test_sim_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("SONIC_DIR", "/models/sonic")
    monkeypatch.setenv("TASK", "portrait-corridors")
    monkeypatch.setenv("IMAGE_WIDTH", "640")
    monkeypatch.setenv("IMAGE_HEIGHT", "480")
    monkeypatch.setenv("JPEG_QUALITY", "85")
    monkeypatch.setenv("VIEWER", "native")
    monkeypatch.setenv("REFERENCE_GHOST", "true")

    assert SimConfig.from_env() == SimConfig(
        device="cpu",
        sonic_dir=Path("/models/sonic"),
        task=get_task("portrait-corridors"),
        image_width=640,
        image_height=480,
        jpeg_quality=85,
        viewer="native",
        reference_ghost=True,
    )


def test_sim_config_accepts_viser_viewer(monkeypatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("SONIC_DIR", "/models/sonic")
    monkeypatch.setenv("TASK", "none")
    monkeypatch.setenv("IMAGE_WIDTH", "640")
    monkeypatch.setenv("IMAGE_HEIGHT", "480")
    monkeypatch.setenv("JPEG_QUALITY", "85")
    monkeypatch.setenv("VIEWER", "viser")
    monkeypatch.setenv("REFERENCE_GHOST", "false")

    assert SimConfig.from_env().viewer == "viser"


def test_sim_config_rejects_unknown_viewer(monkeypatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("SONIC_DIR", "/models/sonic")
    monkeypatch.setenv("TASK", "none")
    monkeypatch.setenv("IMAGE_WIDTH", "640")
    monkeypatch.setenv("IMAGE_HEIGHT", "480")
    monkeypatch.setenv("JPEG_QUALITY", "85")
    monkeypatch.setenv("VIEWER", "unknown")
    monkeypatch.setenv("REFERENCE_GHOST", "false")

    with pytest.raises(ValueError, match="VIEWER"):
        SimConfig.from_env()


def test_sim_config_rejects_invalid_reference_ghost(monkeypatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("SONIC_DIR", "/models/sonic")
    monkeypatch.setenv("TASK", "none")
    monkeypatch.setenv("IMAGE_WIDTH", "640")
    monkeypatch.setenv("IMAGE_HEIGHT", "480")
    monkeypatch.setenv("JPEG_QUALITY", "85")
    monkeypatch.setenv("VIEWER", "native")
    monkeypatch.setenv("REFERENCE_GHOST", "yes")

    with pytest.raises(ValueError, match="REFERENCE_GHOST"):
        SimConfig.from_env()


def test_agent_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("VLM_URL", "http://127.0.0.1:8080/")
    monkeypatch.setenv("VLM_TIMEOUT", "12.5")
    monkeypatch.setenv("VLM_SYSTEM_PROMPT", "/prompts/system.md")
    monkeypatch.setenv("VLM_USER_PROMPT", "/prompts/user.md")

    assert AgentConfig.from_env() == AgentConfig(
        vlm_url="http://127.0.0.1:8080",
        vlm_timeout=12.5,
        system_prompt=Path("/prompts/system.md"),
        user_prompt=Path("/prompts/user.md"),
        command_mode="waypoint",
    )


def test_agent_config_selects_script(monkeypatch) -> None:
    monkeypatch.setenv("AGENT", "script")
    monkeypatch.setenv("SCRIPT_TASK", "box_push")
    monkeypatch.setenv("SCRIPT_PROMPT", "touch the box")

    cfg = AgentConfig.from_env()

    assert cfg.agent == "script"
    assert cfg.script_task == "box_push"
    assert cfg.script_prompt == "touch the box"


def test_agent_config_allows_scripts_to_start_without_observations(monkeypatch) -> None:
    monkeypatch.setenv("AGENT", "script")
    monkeypatch.setenv("SCRIPT_TASK", "box_push")
    monkeypatch.setenv("SCRIPT_PROMPT", "touch the box")
    monkeypatch.setenv("SCRIPT_START_IMMEDIATELY", "true")

    assert AgentConfig.from_env().script_start_immediately is True


def test_script_agent_config_requires_task(monkeypatch) -> None:
    monkeypatch.setenv("AGENT", "script")
    monkeypatch.delenv("SCRIPT_TASK", raising=False)

    with pytest.raises(ValueError, match="SCRIPT_TASK"):
        AgentConfig.from_env()


def test_script_agent_config_requires_prompt(monkeypatch) -> None:
    monkeypatch.setenv("AGENT", "script")
    monkeypatch.setenv("SCRIPT_TASK", "box_push")
    monkeypatch.delenv("SCRIPT_PROMPT", raising=False)

    with pytest.raises(ValueError, match="SCRIPT_PROMPT"):
        AgentConfig.from_env()


def test_missing_runtime_value_fails(monkeypatch) -> None:
    monkeypatch.delenv("PLANNER_ONNX", raising=False)
    monkeypatch.setenv("MOTION_GENERATOR", "kinematic_planner")
    monkeypatch.setenv("DEVICE", "cpu")

    with pytest.raises(KeyError, match="PLANNER_ONNX"):
        MotionGenConfig.from_env()
