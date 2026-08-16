from __future__ import annotations

import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
from dora import Node

from agent.vlm import OAIChatClient
from agent.waypoint import parse_waypoint_command
from shared.arrow import (
    agent_command_to_arrow,
    grounding_request_to_arrow,
    grounding_result_from_arrow,
    observation_from_arrow,
    pipeline_error_from_arrow,
)
from shared.config import AgentConfig
from shared.messages import (
    AgentCommand,
    GroundingRequest,
    GroundingResult,
    PipelineError,
    VisualObservation,
)

MAX_INVALID_RESPONSES = 3
FALLBACK_COMMAND = '{"motion":"stand","waypoints_2d":[]}'
PLANNER_FALLBACK_COMMAND = '{"motion":"stand","direction":"forward"}'


@dataclass(frozen=True)
class _PendingGrounding:
    command_text: str
    motion: str
    waypoints_2d: tuple[tuple[int, int], ...]
    reasoning: str | None


class AgentLoop:
    def __init__(
        self,
        node: Node,
        client: OAIChatClient,
        *,
        waypoint_debug: bool = False,
        command_mode: str = "waypoint",
    ) -> None:
        self.node = node
        self.client = client
        self.observation: VisualObservation | None = None
        self.run_id: int | None = None
        self.pending_command: str | None = None
        self.pending_grounding: _PendingGrounding | None = None
        self.invalid_responses = 0
        self.waypoint_debug = waypoint_debug
        self.command_mode = command_mode

    def run(self) -> None:
        for event in self.node:
            if event["type"] == "STOP":
                return
            if event["type"] != "INPUT":
                continue
            if event["id"] == "observation":
                self._accept_observation(
                    observation_from_arrow(
                        event["value"], dict(event.get("metadata") or {})
                    )
                )
            elif event["id"] in {"planning_error", "sim_error"}:
                self._accept_error(pipeline_error_from_arrow(event["value"]))
            elif event["id"] == "grounding_result":
                self._accept_grounding_result(
                    grounding_result_from_arrow(
                        event["value"], dict(event.get("metadata") or {})
                    )
                )

    def _accept_observation(self, observation: VisualObservation) -> None:
        if observation.run_id != self.run_id:
            if hasattr(self.client, "reset"):
                self.client.reset()
            self.run_id = observation.run_id
            self.observation = None
            self.pending_command = None
            self.pending_grounding = None
            self.invalid_responses = 0
        if self.observation is None:
            if observation.observation_id != 0:
                raise RuntimeError(
                    f"First observation must be 0, got {observation.observation_id}"
                )
            if observation.completed_command is not None:
                raise RuntimeError("Initial observation has a completed command")
        else:
            expected_id = self.observation.observation_id + 1
            if observation.observation_id != expected_id:
                raise RuntimeError(
                    f"Expected observation {expected_id}, got "
                    f"{observation.observation_id}"
                )
            if observation.completed_command != self.pending_command:
                raise RuntimeError(
                    "Completed command does not match the command sent for the "
                    "previous observation"
                )
            assert self.pending_command is not None
            self.client.commit(self.observation, self.pending_command)

        self.observation = observation
        self.pending_command = None
        self.pending_grounding = None
        self.invalid_responses = 0
        self._query_and_send()

    def _accept_error(self, error: PipelineError) -> None:
        if self.observation is None:
            raise RuntimeError(f"{error.source} failed before the first observation")
        if error.observation_id != self.observation.observation_id:
            raise RuntimeError(
                f"Stale {error.source} error for observation {error.observation_id}"
            )
        if error.source == "grounding":
            if self.pending_grounding is None:
                raise RuntimeError("Grounding failed without a pending request")
            previous = self.pending_grounding.command_text
            self.pending_grounding = None
            self._retry_invalid(previous, error.detail)
            return
        if error.source != "motion-gen":
            self.node.log(
                "error",
                f"[OBS {error.observation_id}] {error.source} error: {error.detail}",
                target="dsrf.agent",
                fields={
                    "event": "pipeline_error",
                    "observation_id": str(error.observation_id),
                    "source": error.source,
                    "detail": error.detail,
                },
            )
            raise RuntimeError(f"{error.source}: {error.detail}")

        previous = self.pending_command or ""
        self._retry_invalid(previous, error.detail)

    def _retry_invalid(self, previous: str, detail: str) -> None:
        assert self.observation is not None
        self.pending_grounding = None
        self.invalid_responses += 1
        observation_id = self.observation.observation_id
        self.node.log(
            "warn",
            f"[OBS {observation_id}] invalid command: {previous!r} error={detail!r}",
            target="dsrf.agent",
            fields={
                "event": "invalid_command",
                "observation_id": str(observation_id),
                "command": previous,
                "detail": detail,
                "attempt": str(self.invalid_responses),
            },
        )
        if self.invalid_responses >= MAX_INVALID_RESPONSES:
            self.node.log(
                "warn",
                f"[OBS {observation_id}] fallback command: "
                f"{self._fallback_command!r} after {self.invalid_responses} invalid responses",
                target="dsrf.agent",
                fields={
                    "event": "fallback_command",
                    "observation_id": str(observation_id),
                    "command": self._fallback_command,
                    "invalid_responses": str(self.invalid_responses),
                },
            )
            self._send(self._fallback_command, motion="stand", target_xys=())
            return
        feedback = f"Your previous response {previous!r} was invalid: {detail}"
        self._query_and_send(retry_feedback=feedback)

    def _query_and_send(self, *, retry_feedback: str | None = None) -> None:
        assert self.observation is not None
        observation_id = self.observation.observation_id
        attempt = self.invalid_responses
        fields = {
            "event": "vlm_request",
            "observation_id": str(observation_id),
            "attempt": str(attempt),
        }
        self.node.log(
            "debug",
            f"[OBS {observation_id}] VLM request started retry={attempt}",
            target="dsrf.agent.vlm",
            fields=fields,
        )
        started_at = time.perf_counter()
        try:
            result = self.client.complete(
                self.observation,
                retry_feedback=retry_feedback,
            )
        except Exception as exc:
            vlm_ms = (time.perf_counter() - started_at) * 1000.0
            detail = f"{type(exc).__name__}: {exc}"
            self.node.log(
                "error",
                f"[OBS {observation_id}] VLM request failed: {detail}",
                target="dsrf.agent.vlm",
                fields={
                    "event": "vlm_error",
                    "observation_id": str(observation_id),
                    "attempt": str(attempt),
                    "vlm_ms": f"{vlm_ms:.1f}",
                    "detail": detail,
                },
            )
            raise

        vlm_ms = (time.perf_counter() - started_at) * 1000.0
        command = str(result)
        reasoning = getattr(result, "reasoning", None)
        self.node.log(
            "info",
            f"[OBS {observation_id}] VLM command: {command!r} "
            f"vlm_ms={vlm_ms:.1f} retry={attempt}",
            target="dsrf.agent.vlm",
            fields={
                "event": "vlm_response",
                "observation_id": str(observation_id),
                "command": command,
                "vlm_ms": f"{vlm_ms:.1f}",
                "attempt": str(attempt),
                "jpeg_kb": f"{len(self.observation.jpeg) / 1024.0:.1f}",
            },
        )
        try:
            if self.command_mode == "direction":
                if not _is_waypoint_command(command):
                    motion, direction = _parse_planner_command(command)
                    self._send(
                        command,
                        motion=motion,
                        target_xys=(),
                        direction=direction,
                        reasoning=reasoning,
                    )
                    return
            parsed = parse_waypoint_command(command)
            if not parsed.waypoints_2d:
                self._send(
                    command, motion=parsed.motion, target_xys=(), reasoning=reasoning
                )
                return
            request = GroundingRequest(observation_id, parsed.waypoints_2d)
        except ValueError as exc:
            self._retry_invalid(command, str(exc))
            return

        self.pending_grounding = _PendingGrounding(
            command_text=command,
            motion=parsed.motion,
            waypoints_2d=parsed.waypoints_2d,
            reasoning=reasoning,
        )
        data, metadata = grounding_request_to_arrow(request)
        self.node.send_output("grounding_request", data, metadata=metadata)

    def _accept_grounding_result(self, result: GroundingResult) -> None:
        if self.observation is None or self.pending_grounding is None:
            raise RuntimeError("Received an unexpected grounding result")
        if result.observation_id != self.observation.observation_id:
            raise RuntimeError(
                f"Stale grounding result for observation {result.observation_id}"
            )
        pending = self.pending_grounding
        self.pending_grounding = None
        if self.waypoint_debug:
            self.node.log(
                "info",
                f"VLM waypoints: normalized={pending.waypoints_2d} "
                f"local_targets={result.target_xys}",
                target="dsrf.agent.waypoint",
                fields={
                    "event": "waypoint_grounded",
                    "observation_id": str(result.observation_id),
                },
            )
            _write_debug_image(
                self.observation.jpeg,
                result.observation_id,
                pending.waypoints_2d,
            )
        self._send(
            pending.command_text,
            motion=pending.motion,
            target_xys=result.target_xys,
            reasoning=pending.reasoning,
        )

    def _send(
        self,
        command_text: str,
        *,
        motion: str,
        target_xys: tuple[tuple[float, float], ...],
        direction: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        assert self.observation is not None
        command = AgentCommand(
            self.observation.observation_id,
            command_text,
            motion,
            target_xys,
            direction,
            reasoning,
        )
        data, metadata = agent_command_to_arrow(command)
        self.node.send_output("command", data, metadata=metadata)
        self.pending_command = command.text

    @property
    def _fallback_command(self) -> str:
        return (
            PLANNER_FALLBACK_COMMAND
            if self.command_mode == "direction"
            else FALLBACK_COMMAND
        )


def _write_debug_image(
    jpeg: bytes, observation_id: int, waypoints_2d: tuple[tuple[int, int], ...]
) -> None:
    image = iio.imread(BytesIO(jpeg), extension=".jpg")
    height, width = image.shape[:2]
    for x, y in waypoints_2d:
        u = round(x / 1000 * (width - 1))
        v = round(y / 1000 * (height - 1))
        image[max(0, v - 5) : v + 6, u] = (255, 0, 0)
        image[v, max(0, u - 5) : u + 6] = (255, 0, 0)
    output_dir = Path("/tmp/dsrf-waypoint-debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_dir / f"observation-{observation_id}.jpg", image)


def main() -> None:
    cfg = AgentConfig.from_env()
    node = Node()
    client = OAIChatClient(
        base_url=cfg.vlm_url,
        timeout=cfg.vlm_timeout,
        system_prompt=cfg.system_prompt.read_text(encoding="utf-8"),
        user_prompt=cfg.user_prompt.read_text(encoding="utf-8"),
        history_turns=cfg.history_turns,
        history_retain_turns=cfg.history_retain_turns,
    )
    AgentLoop(
        node,
        client,
        waypoint_debug=cfg.waypoint_debug,
        command_mode=cfg.command_mode,
    ).run()


def _parse_planner_command(text: str) -> tuple[str, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Command must be a JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"motion", "direction"}:
        raise ValueError("Planner command must contain only motion and direction")
    motion = payload["motion"]
    direction = payload["direction"]
    if motion == "stand" and direction in {"forward", "backward", "left", "right"}:
        return "stand", None
    if motion == "walk" and direction in {"forward", "backward", "left", "right"}:
        return "walk", direction
    if motion == "turn" and direction in {"left", "right"}:
        return "turn", direction
    raise ValueError("Unsupported planner motion or direction")


def _is_waypoint_command(text: str) -> bool:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and "waypoints_2d" in payload


if __name__ == "__main__":
    main()
