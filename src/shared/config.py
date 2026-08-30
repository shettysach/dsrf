from __future__ import annotations

import math
import os
from dataclasses import MISSING, dataclass, fields
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeAliasType,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from tasks import TaskSpec, get_task

type ViewerMode = Literal["none", "native", "viser"]


@dataclass(frozen=True)
class MotionGenConfig:
    device: str
    backend: KinematicPlannerConfig | ArdyConfig

    @classmethod
    def from_env(cls) -> "MotionGenConfig":
        backend = (
            ArdyConfig.from_env()
            if _motion_generator() == "ardy"
            else KinematicPlannerConfig.from_env()
        )
        return _dataclass_from_env(cls, overrides={"backend": backend})


@dataclass(frozen=True)
class KinematicPlannerConfig:
    planner_onnx: Path

    @classmethod
    def from_env(cls) -> "KinematicPlannerConfig":
        return _dataclass_from_env(cls)


@dataclass(frozen=True)
class ArdyConfig:
    checkpoints_dir: Path
    text_encoder_model: Path
    text_encoder_device: str

    @classmethod
    def from_env(cls) -> "ArdyConfig":
        return _dataclass_from_env(cls)


@dataclass(frozen=True)
class SimConfig:
    device: str
    sonic_dir: Path
    task: TaskSpec | None
    image_width: int
    image_height: int
    jpeg_quality: int
    viewer: ViewerMode
    reference_ghost: bool
    publish_observations: bool = True
    demo_video_path: Path | None = None
    stop_on_stand: bool = False
    demo_max_commands: int | None = None
    demo_timeout_seconds: float | None = None

    @classmethod
    def from_env(cls) -> "SimConfig":
        return _dataclass_from_env(
            cls,
            overrides={
                "task": _optional_task(),
                "publish_observations": _optional_boolean(
                    "PUBLISH_OBSERVATIONS", default=True
                ),
                "demo_video_path": _optional_path("DEMO_VIDEO_PATH"),
                "stop_on_stand": _optional_boolean("STOP_ON_STAND", default=False),
                "demo_max_commands": _optional_positive_int("DEMO_MAX_COMMANDS"),
                "demo_timeout_seconds": _optional_positive_float(
                    "DEMO_TIMEOUT_SECONDS"
                ),
            },
        )

    def __post_init__(self) -> None:
        for name, value in (
            ("IMAGE_WIDTH", self.image_width),
            ("IMAGE_HEIGHT", self.image_height),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"JPEG_QUALITY must be in 1..100, got {self.jpeg_quality}")


@dataclass(frozen=True)
class AgentConfig:
    vlm_url: str
    vlm_timeout: float
    system_prompt: Path = dataclass_field(metadata={"env": "VLM_SYSTEM_PROMPT"})
    user_prompt: Path = dataclass_field(metadata={"env": "VLM_USER_PROMPT"})
    command_mode: Literal["waypoint", "direction"]
    max_vlm_turns: int | None = None
    agent: Literal["vlm", "script"] = "vlm"
    script_task: str | None = None
    script_prompt: str | None = None
    script_start_immediately: bool = False

    @classmethod
    def from_env(cls) -> "AgentConfig":
        agent = os.environ.get("AGENT", "vlm").strip().lower()
        if agent not in {"vlm", "script"}:
            raise ValueError("AGENT must be 'vlm' or 'script'")
        command_mode: Literal["waypoint", "direction"] = (
            "direction"
            if os.environ.get("MOTION_GENERATOR", "").strip().lower()
            == "kinematic_planner"
            else "waypoint"
        )
        overrides: dict[str, object] = {
            "command_mode": command_mode,
            "max_vlm_turns": _optional_positive_int("DEMO_MAX_COMMANDS"),
            "agent": agent,
            "script_task": _optional_env("SCRIPT_TASK"),
            "script_prompt": _optional_env("SCRIPT_PROMPT"),
            "script_start_immediately": _optional_boolean(
                "SCRIPT_START_IMMEDIATELY", default=False
            ),
        }
        if agent == "script":
            # Script mode never instantiates the VLM client, so it should be
            # runnable without endpoint or prompt configuration.
            overrides.update(
                vlm_url="",
                vlm_timeout=1.0,
                system_prompt=Path(),
                user_prompt=Path(),
            )
        return _dataclass_from_env(cls, overrides=overrides)

    def __post_init__(self) -> None:
        if self.agent not in {"vlm", "script"}:
            raise ValueError("AGENT must be 'vlm' or 'script'")
        if self.agent == "script" and self.script_task is None:
            raise ValueError("SCRIPT_TASK must be set when AGENT is 'script'")
        if self.agent == "script" and self.script_prompt is None:
            raise ValueError("SCRIPT_PROMPT must be set when AGENT is 'script'")
        if self.agent == "vlm":
            url = self.vlm_url.strip().rstrip("/")
            if not url:
                raise ValueError("VLM_URL must not be empty")
            if not math.isfinite(self.vlm_timeout) or self.vlm_timeout <= 0.0:
                raise ValueError("VLM_TIMEOUT must be positive")
            object.__setattr__(self, "vlm_url", url)


def _optional_name(name: str) -> str | None:
    value = os.environ[name].strip()
    return None if value.lower() == "none" or not value else value


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _optional_task() -> TaskSpec | None:
    name = _optional_name("TASK")
    return get_task(name) if name is not None else None


def _optional_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _optional_boolean(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in {"false", "true"}:
        raise ValueError(f"{name} must be 'false' or 'true'")
    return value == "true"


def _optional_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_positive_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _motion_generator() -> Literal["kinematic_planner", "ardy"]:
    value = os.environ["MOTION_GENERATOR"].strip().lower()
    if value not in {"kinematic_planner", "ardy"}:
        raise ValueError("MOTION_GENERATOR must be 'kinematic_planner' or 'ardy'")
    return value


def _dataclass_from_env[T](
    config_type: type[T],
    *,
    prefix: str = "",
    overrides: dict[str, object] | None = None,
) -> T:
    """Construct a dataclass from environment values and computed fields."""
    type_hints = get_type_hints(config_type)
    values = dict(overrides or {})
    for field in fields(cast(Any, config_type)):
        if field.name in values:
            continue
        env_name = str(field.metadata.get("env", f"{prefix}{field.name.upper()}"))
        if env_name in os.environ:
            values[field.name] = _parse_env_value(
                env_name,
                os.environ[env_name],
                type_hints[field.name],
            )
        elif field.default is MISSING and field.default_factory is MISSING:
            raise KeyError(env_name)
    return config_type(**values)


def _parse_env_value(name: str, value: str, value_type: object) -> object:
    if isinstance(value_type, TypeAliasType):
        return _parse_env_value(name, value, value_type.__value__)
    if value_type is str:
        return value
    if value_type is float:
        return float(value)
    if value_type is int:
        return int(value)
    if value_type is bool:
        if value not in {"false", "true"}:
            raise ValueError(f"{name} must be 'false' or 'true'")
        return value == "true"
    if value_type is Path:
        return Path(value)
    if get_origin(value_type) is Literal:
        choices = get_args(value_type)
        if value not in choices:
            expected = ", ".join(repr(choice) for choice in choices)
            raise ValueError(f"{name} must be one of {expected}")
        return value
    raise TypeError(f"Unsupported environment-backed config type for {name}")
