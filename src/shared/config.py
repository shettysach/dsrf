from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tasks import TaskSpec, get_task

type ViewerMode = Literal["none", "native", "viser"]


@dataclass(frozen=True)
class MotionGenConfig:
    device: str
    backend: KinematicPlannerConfig | ArdyConfig

    @classmethod
    def from_env(cls) -> "MotionGenConfig":
        generator = _motion_generator()
        if generator == "ardy":
            return cls(
                device=os.environ["DEVICE"],
                backend=ArdyConfig(
                    checkpoints_dir=Path(os.environ["CHECKPOINTS_DIR"]),
                    text_encoder_model=Path(os.environ["TEXT_ENCODER_MODEL"]),
                    text_encoder_device=os.environ["TEXT_ENCODER_DEVICE"],
                ),
            )
        return cls(
            device=os.environ["DEVICE"],
            backend=KinematicPlannerConfig(
                planner_onnx=Path(os.environ["PLANNER_ONNX"]),
            ),
        )


@dataclass(frozen=True)
class KinematicPlannerConfig:
    planner_onnx: Path


@dataclass(frozen=True)
class ArdyConfig:
    checkpoints_dir: Path
    text_encoder_model: Path
    text_encoder_device: str


@dataclass(frozen=True)
class SimConfig:
    sonic_dir: Path
    device: str
    task: TaskSpec | None
    image_width: int
    image_height: int
    jpeg_quality: int
    capture_depth: bool
    viewer: ViewerMode
    reference_ghost: bool
    demo_video_path: Path | None = None
    demo_video_dir: Path | None = None
    demo_runs: int = 1
    motion_timeout_seconds: float = 20.0
    goal_index: int | None = None

    @classmethod
    def from_env(cls) -> "SimConfig":
        return cls(
            sonic_dir=Path(os.environ["SONIC_DIR"]),
            device=os.environ["DEVICE"],
            task=_optional_task(),
            image_width=_positive_int("IMAGE_WIDTH"),
            image_height=_positive_int("IMAGE_HEIGHT"),
            jpeg_quality=_bounded_int("JPEG_QUALITY", minimum=1, maximum=100),
            capture_depth=_optional_boolean("CAPTURE_DEPTH", default=True),
            viewer=_viewer_mode(),
            reference_ghost=_boolean("REFERENCE_GHOST"),
            demo_video_path=(
                Path(value)
                if (value := os.environ.get("DEMO_VIDEO_PATH", "").strip())
                else None
            ),
            demo_video_dir=(
                Path(value)
                if (value := os.environ.get("DEMO_VIDEO_DIR", "").strip())
                else None
            ),
            demo_runs=_positive_int_default("DEMO_RUNS", default=1),
            motion_timeout_seconds=_positive_float(
                "MOTION_TIMEOUT_SECONDS", default=20.0
            ),
            goal_index=_optional_goal_index(),
        )


@dataclass(frozen=True)
class AgentConfig:
    vlm_url: str
    vlm_timeout: float
    system_prompt: Path
    user_prompt: Path
    waypoint_debug: bool
    command_mode: Literal["waypoint", "direction"]

    @classmethod
    def from_env(cls) -> "AgentConfig":
        url = os.environ["VLM_URL"].strip().rstrip("/")
        if not url:
            raise ValueError("VLM_URL must not be empty")
        timeout = float(os.environ["VLM_TIMEOUT"])
        if timeout <= 0.0:
            raise ValueError("VLM_TIMEOUT must be positive")
        return cls(
            vlm_url=url,
            vlm_timeout=timeout,
            system_prompt=Path(os.environ["VLM_SYSTEM_PROMPT"]),
            user_prompt=Path(os.environ["VLM_USER_PROMPT"]),
            waypoint_debug=_optional_boolean("WAYPOINT_DEBUG", default=False),
            command_mode=(
                "direction"
                if os.environ.get("MOTION_GENERATOR", "").strip().lower()
                == "kinematic_planner"
                else "waypoint"
            ),
        )


def _positive_int(name: str) -> int:
    return _bounded_int(name, minimum=1)


def _positive_int_default(name: str, *, default: int) -> int:
    if name not in os.environ:
        return default
    return _positive_int(name)


def _positive_float(name: str, *, default: float | None = None) -> float:
    value = float(os.environ[name]) if default is None or name in os.environ else default
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_goal_index() -> int | None:
    value = os.environ.get("GOAL_INDEX", "").strip()
    if not value:
        return None
    index = int(value)
    if index not in {0, 1, 2}:
        raise ValueError(f"GOAL_INDEX must be 0, 1, or 2, got {index}")
    return index


def _bounded_int(name: str, *, minimum: int, maximum: int | None = None) -> int:
    value = int(os.environ[name])
    if value < minimum or (maximum is not None and value > maximum):
        expected = f">= {minimum}" if maximum is None else f"{minimum}..{maximum}"
        raise ValueError(f"{name} must be in {expected}, got {value}")
    return value


def _optional_name(name: str) -> str | None:
    value = os.environ[name].strip()
    return None if value.lower() == "none" or not value else value


def _optional_task() -> TaskSpec | None:
    name = _optional_name("TASK")
    return get_task(name) if name is not None else None


def _viewer_mode() -> ViewerMode:
    value = os.environ["VIEWER"].strip().lower()
    if value not in {"none", "native", "viser"}:
        raise ValueError("VIEWER must be 'none', 'native', or 'viser'")
    return value


def _motion_generator() -> Literal["kinematic_planner", "ardy"]:
    value = os.environ["MOTION_GENERATOR"].strip().lower()
    if value not in {"kinematic_planner", "ardy"}:
        raise ValueError("MOTION_GENERATOR must be 'kinematic_planner' or 'ardy'")
    return value


def _boolean(name: str) -> bool:
    value = os.environ[name].strip().lower()
    if value not in {"false", "true"}:
        raise ValueError(f"{name} must be 'false' or 'true'")
    return value == "true"


def _optional_boolean(name: str, *, default: bool) -> bool:
    if name not in os.environ:
        return default
    return _boolean(name)
