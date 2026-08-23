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
    controller: SonicConfig | VirtualForcesConfig
    task: TaskSpec | None
    image_width: int
    image_height: int
    jpeg_quality: int
    viewer: ViewerMode
    reference_ghost: bool

    @classmethod
    def from_env(cls) -> "SimConfig":
        return _dataclass_from_env(
            cls,
            overrides={
                "controller": _controller(),
                "task": _optional_task(),
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
class SonicConfig:
    sonic_dir: Path

    @classmethod
    def from_env(cls) -> "SonicConfig":
        return _dataclass_from_env(cls)


@dataclass(frozen=True)
class VirtualForcesConfig:
    assistance_enabled: bool = False
    target_body: str = "pelvis"
    position_kp: float = 400.0
    position_kd: float = 40.0
    orientation_kp: float = 100.0
    orientation_kd: float = 10.0
    position_deadband: float = 0.02
    orientation_deadband: float = 0.05
    force_limit: float = 300.0
    torque_limit: float = 100.0

    @classmethod
    def from_env(cls) -> "VirtualForcesConfig":
        return _dataclass_from_env(cls, prefix="VF_")

    def __post_init__(self) -> None:
        if not self.target_body.strip():
            raise ValueError("VF_TARGET_BODY must not be empty")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and (not math.isfinite(value) or value < 0.0):
                name = f"VF_{field.name.upper()}"
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class AgentConfig:
    vlm_url: str
    vlm_timeout: float
    system_prompt: Path = dataclass_field(metadata={"env": "VLM_SYSTEM_PROMPT"})
    user_prompt: Path = dataclass_field(metadata={"env": "VLM_USER_PROMPT"})
    command_mode: Literal["waypoint", "direction"]

    @classmethod
    def from_env(cls) -> "AgentConfig":
        command_mode: Literal["waypoint", "direction"] = (
            "direction"
            if os.environ.get("MOTION_GENERATOR", "").strip().lower()
            == "kinematic_planner"
            else "waypoint"
        )
        return _dataclass_from_env(cls, overrides={"command_mode": command_mode})

    def __post_init__(self) -> None:
        url = self.vlm_url.strip().rstrip("/")
        if not url:
            raise ValueError("VLM_URL must not be empty")
        if not math.isfinite(self.vlm_timeout) or self.vlm_timeout <= 0.0:
            raise ValueError("VLM_TIMEOUT must be positive")
        object.__setattr__(self, "vlm_url", url)


def _optional_name(name: str) -> str | None:
    value = os.environ[name].strip()
    return None if value.lower() == "none" or not value else value


def _optional_task() -> TaskSpec | None:
    name = _optional_name("TASK")
    return get_task(name) if name is not None else None


def _motion_generator() -> Literal["kinematic_planner", "ardy"]:
    value = os.environ["MOTION_GENERATOR"].strip().lower()
    if value not in {"kinematic_planner", "ardy"}:
        raise ValueError("MOTION_GENERATOR must be 'kinematic_planner' or 'ardy'")
    return value


def _controller() -> SonicConfig | VirtualForcesConfig:
    value = os.environ["CONTROLLER"].strip().lower()
    if value == "sonic":
        return SonicConfig.from_env()
    if value == "virtual_forces":
        return VirtualForcesConfig.from_env()
    raise ValueError("CONTROLLER must be 'sonic' or 'virtual_forces'")


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
