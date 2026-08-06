from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pyarrow as pa

MOTION_COLUMNS = 36
SONIC_FPS = 50
ARDY_EMBEDDING_SIZE = 4096


@dataclass(frozen=True)
class ProjectionContext:
    depth: np.ndarray
    camera_pos_w: np.ndarray
    camera_forward_w: np.ndarray
    camera_up_w: np.ndarray
    frustum_height: float
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    near: float
    far: float

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float32)
        if depth.ndim != 2 or not depth.size:
            raise ValueError(f"Depth must have shape [H, W], got {depth.shape}")
        vectors = {
            "camera_pos_w": (self.camera_pos_w, (3,)),
            "camera_forward_w": (self.camera_forward_w, (3,)),
            "camera_up_w": (self.camera_up_w, (3,)),
            "root_pos_w": (self.root_pos_w, (3,)),
            "root_quat_w": (self.root_quat_w, (4,)),
        }
        for name, (value, shape) in vectors.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, np.ascontiguousarray(array))
        if not np.isfinite(self.frustum_height) or self.frustum_height <= 0.0:
            raise ValueError("Camera frustum height must be positive and finite")
        if not np.isfinite(self.near) or not np.isfinite(self.far):
            raise ValueError("Camera clipping distances must be finite")
        if self.near <= 0.0 or self.far <= self.near:
            raise ValueError("Camera clipping distances are invalid")
        object.__setattr__(self, "depth", np.ascontiguousarray(depth))


@dataclass(frozen=True)
class MotionChunk:
    observation_id: int
    command: str
    qpos: np.ndarray

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if not self.command.strip():
            raise ValueError("Motion command is empty")
        qpos = np.asarray(self.qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != MOTION_COLUMNS:
            raise ValueError(
                f"Motion qpos must have shape [T, {MOTION_COLUMNS}], got {qpos.shape}"
            )
        if qpos.shape[0] == 0:
            raise ValueError("Motion chunk must contain at least one frame")
        if not np.isfinite(qpos).all():
            raise ValueError("Motion chunk contains NaN or infinite values")
        object.__setattr__(self, "qpos", np.ascontiguousarray(qpos))


@dataclass(frozen=True)
class AgentCommand:
    observation_id: int
    text: str
    motion: str
    target_xy: tuple[float, float] | None
    direction: str | None = None

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Command is empty")
        _validate_navigation(self.motion, self.target_xy, self.direction)
        object.__setattr__(self, "text", normalized)


@dataclass(frozen=True)
class EncodedCommand:
    observation_id: int
    text: str
    motion: str
    target_xy: tuple[float, float] | None
    embedding: np.ndarray
    direction: str | None = None

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        normalized = self.text.strip()
        if not normalized:
            raise ValueError("Command is empty")
        _validate_navigation(self.motion, self.target_xy, self.direction)
        embedding = np.asarray(self.embedding, dtype=np.float32)
        if embedding.shape != (ARDY_EMBEDDING_SIZE,):
            raise ValueError(
                "Command embedding must have shape "
                f"[{ARDY_EMBEDDING_SIZE}], got {embedding.shape}"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("Command embedding contains NaN or infinite values")
        object.__setattr__(self, "text", normalized)
        object.__setattr__(self, "embedding", np.ascontiguousarray(embedding))


@dataclass(frozen=True)
class VisualObservation:
    observation_id: int
    completed_command: str | None
    jpeg: bytes
    projection: ProjectionContext | None = None

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if not self.jpeg:
            raise ValueError("Observation JPEG is empty")


@dataclass(frozen=True)
class PipelineError:
    source: str
    observation_id: int
    detail: str

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("Observation ID must be non-negative")
        if not self.source:
            raise ValueError("Error source is empty")
        if not self.detail:
            raise ValueError("Error detail is empty")


def motion_to_arrow(chunk: MotionChunk) -> tuple[pa.Array, dict[str, str]]:
    return pa.array(chunk.qpos.reshape(-1), type=pa.float32()), {
        "observation_id": str(chunk.observation_id),
        "command": chunk.command,
    }


def motion_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> MotionChunk:
    flat = np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32)
    if flat.size == 0 or flat.size % MOTION_COLUMNS:
        raise ValueError(
            f"Motion payload has {flat.size} values; expected complete "
            f"{MOTION_COLUMNS}-value frames"
        )
    return MotionChunk(
        observation_id=_observation_id(metadata),
        command=str(metadata["command"]),
        qpos=flat.reshape(-1, MOTION_COLUMNS),
    )


def agent_command_to_arrow(
    command: AgentCommand,
) -> tuple[pa.Array, dict[str, str]]:
    metadata = {
        "observation_id": str(command.observation_id),
        "motion": command.motion,
    }
    if command.target_xy is not None:
        metadata["target_xy"] = json.dumps(command.target_xy, separators=(",", ":"))
    if command.direction is not None:
        metadata["direction"] = command.direction
    return pa.array([command.text], type=pa.string()), metadata


def agent_command_from_arrow(value: pa.Array, metadata: dict[str, Any]) -> AgentCommand:
    return AgentCommand(
        observation_id=_observation_id(metadata),
        text=_string_from_arrow(value),
        motion=str(metadata["motion"]),
        target_xy=_target_xy(metadata),
        direction=_direction(metadata),
    )


def encoded_command_to_arrow(
    command: EncodedCommand,
) -> tuple[pa.Array, dict[str, str]]:
    return pa.array(command.embedding, type=pa.float32()), {
        "observation_id": str(command.observation_id),
        "text": command.text,
        "motion": command.motion,
        **(
            {"target_xy": json.dumps(command.target_xy, separators=(",", ":"))}
            if command.target_xy is not None
            else {}
        ),
        **({"direction": command.direction} if command.direction is not None else {}),
    }


def encoded_command_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> EncodedCommand:
    return EncodedCommand(
        observation_id=_observation_id(metadata),
        text=str(metadata["text"]),
        motion=str(metadata["motion"]),
        target_xy=_target_xy(metadata),
        embedding=np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float32),
        direction=_direction(metadata),
    )


def observation_to_arrow(
    observation: VisualObservation,
) -> tuple[pa.Array, dict[str, str]]:
    metadata = {
        "observation_id": str(observation.observation_id),
        "mime_type": "image/jpeg",
    }
    if observation.completed_command is not None:
        metadata["completed_command"] = observation.completed_command
    depth: bytes | None = None
    if observation.projection is not None:
        projection = observation.projection
        depth = projection.depth.tobytes()
        metadata["projection"] = json.dumps(
            {
                "shape": projection.depth.shape,
                "camera_pos_w": projection.camera_pos_w.tolist(),
                "camera_forward_w": projection.camera_forward_w.tolist(),
                "camera_up_w": projection.camera_up_w.tolist(),
                "frustum_height": projection.frustum_height,
                "root_pos_w": projection.root_pos_w.tolist(),
                "root_quat_w": projection.root_quat_w.tolist(),
                "near": projection.near,
                "far": projection.far,
            },
            separators=(",", ":"),
        )
    value = pa.array(
        [{"jpeg": observation.jpeg, "depth": depth}],
        type=pa.struct([("jpeg", pa.binary()), ("depth", pa.binary())]),
    )
    return value, metadata


def observation_from_arrow(
    value: pa.Array, metadata: dict[str, Any]
) -> VisualObservation:
    mime_type = metadata.get("mime_type")
    if mime_type != "image/jpeg":
        raise ValueError(f"Unsupported observation MIME type: {mime_type!r}")
    rows = value.to_pylist()
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("Expected one observation struct")
    jpeg = rows[0].get("jpeg")
    depth_bytes = rows[0].get("depth")
    if not isinstance(jpeg, bytes):
        raise ValueError("Observation JPEG is invalid")
    projection = None
    if "projection" in metadata:
        if not isinstance(depth_bytes, bytes):
            raise ValueError("Observation depth is invalid")
        document = json.loads(str(metadata["projection"]))
        shape = tuple(int(size) for size in document["shape"])
        depth = np.frombuffer(depth_bytes, dtype=np.float32).copy().reshape(shape)
        projection = ProjectionContext(
            depth=depth,
            camera_pos_w=document["camera_pos_w"],
            camera_forward_w=document["camera_forward_w"],
            camera_up_w=document["camera_up_w"],
            frustum_height=float(document["frustum_height"]),
            root_pos_w=document["root_pos_w"],
            root_quat_w=document["root_quat_w"],
            near=float(document["near"]),
            far=float(document["far"]),
        )
    return VisualObservation(
        observation_id=_observation_id(metadata),
        completed_command=(
            str(metadata["completed_command"])
            if "completed_command" in metadata
            else None
        ),
        jpeg=jpeg,
        projection=projection,
    )


def pipeline_error_to_arrow(error: PipelineError) -> pa.Array:
    return _json_to_arrow(asdict(error))


def pipeline_error_from_arrow(value: pa.Array) -> PipelineError:
    data = _json_from_arrow(value)
    return PipelineError(
        source=str(data["source"]),
        observation_id=int(data["observation_id"]),
        detail=str(data["detail"]),
    )


def _json_to_arrow(value: dict[str, Any]) -> pa.Array:
    return pa.array([json.dumps(value, separators=(",", ":"))], type=pa.string())


def _json_from_arrow(value: pa.Array) -> dict[str, Any]:
    decoded = json.loads(_string_from_arrow(value))
    if not isinstance(decoded, dict):
        raise ValueError("Expected a JSON object")
    return decoded


def _string_from_arrow(value: pa.Array) -> str:
    values = value.to_pylist()
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("Expected one string")
    return values[0]


def _binary_from_arrow(value: pa.Array) -> bytes:
    values = value.to_pylist()
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise ValueError("Expected one binary value")
    return values[0]


def _observation_id(metadata: dict[str, Any]) -> int:
    observation_id = int(metadata["observation_id"])
    if observation_id < 0:
        raise ValueError("Observation ID must be non-negative")
    return observation_id


def _target_xy(metadata: dict[str, Any]) -> tuple[float, float] | None:
    if "target_xy" not in metadata:
        return None
    value = json.loads(str(metadata["target_xy"]))
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("target_xy metadata must contain two values")
    return (float(value[0]), float(value[1]))


def _direction(metadata: dict[str, Any]) -> str | None:
    if "direction" not in metadata:
        return None
    direction = str(metadata["direction"])
    if direction not in {"forward", "backward", "left", "right"}:
        raise ValueError("Unsupported direction")
    return direction


def _validate_navigation(
    motion: str, target_xy: tuple[float, float] | None, direction: str | None
) -> None:
    if motion not in {"stand", "walk"}:
        raise ValueError("Motion must be stand or walk")
    if motion == "stand":
        if target_xy is not None or direction is not None:
            raise ValueError("Stand command must not have a target")
        return
    if (target_xy is None) == (direction is None):
        raise ValueError("Walk command requires exactly one target")
    if direction is not None:
        if direction not in {"forward", "backward", "left", "right"}:
            raise ValueError("Unsupported direction")
        return
    assert target_xy is not None
    if not all(np.isfinite(value) for value in target_xy):
        raise ValueError("target_xy must be finite")
