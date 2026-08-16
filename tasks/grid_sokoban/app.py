from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pygame

from agent.vlm import OAIChatClient
from shared.messages import VisualObservation
from tasks.grid_sokoban.env import GridSokoban, available_layouts, make_layout
from tasks.grid_sokoban.protocol import parse_action

CELL_PIXELS = 72
PANEL_HEIGHT = 104
DEBUG_PANEL_WIDTH = 420
FPS = 60
AUTO_DELAY_SECONDS = 0.2
MAX_INVALID_VLM_RESPONSES = 3
BACKGROUND = (24, 29, 38)
FLOOR = (235, 229, 213)
GRID = (205, 198, 181)
WALL = (65, 72, 82)
GOAL = (75, 176, 104)
BOX = (229, 163, 49)
BOX_EDGE = (170, 111, 25)
PLAYER = (67, 135, 222)
TEXT = (238, 242, 248)
MUTED_TEXT = (175, 185, 200)


@dataclass(frozen=True)
class _VlmResponse:
    output: str
    reasoning: str | None


class SokobanApp:
    def __init__(
        self, *, layout_name: str, vlm: OAIChatClient | None, auto_play: bool
    ) -> None:
        self.layout_name = layout_name
        self.board = GridSokoban(make_layout(layout_name))
        self.vlm = vlm
        self.auto_play = auto_play and vlm is not None
        self.moves = 0
        self.status = "Arrow keys / WASD to move · R to reset"
        self.observation_id = 0
        self.previous_observation: VisualObservation | None = None
        self.pending_command: str | None = None
        self._previous_turn_committed = False
        self._invalid_response_count = 0
        self.last_auto_at = time.monotonic()
        self._vlm_request_id = 0
        self._vlm_in_flight = False
        self._vlm_results: queue.SimpleQueue[
            tuple[int, VisualObservation, _VlmResponse | Exception]
        ] = queue.SimpleQueue()
        self.last_vlm_output = "No VLM response yet."
        self.last_vlm_reasoning = "No reasoning field returned yet."
        self.last_retry_feedback = "None"
        pygame.init()
        pygame.display.set_caption("Grid Sokoban")
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        window_size = (
            self.board.cols * CELL_PIXELS + DEBUG_PANEL_WIDTH,
            self.board.rows * CELL_PIXELS + PANEL_HEIGHT,
        )
        self.window = pygame.display.set_mode(window_size)
        self.board_surface = pygame.Surface(
            (self.board.cols * CELL_PIXELS, window_size[1] - PANEL_HEIGHT)
        )

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        self._draw()
        pygame.display.flip()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
            self._consume_vlm_result()
            if (
                self.auto_play
                and not self.board.solved
                and not self._vlm_in_flight
                and time.monotonic() - self.last_auto_at >= AUTO_DELAY_SECONDS
            ):
                self._start_vlm_request()
            self._draw()
            pygame.display.flip()
            clock.tick(FPS)
        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        key_actions = {
            pygame.K_UP: "up",
            pygame.K_w: "up",
            pygame.K_DOWN: "down",
            pygame.K_s: "down",
            pygame.K_LEFT: "left",
            pygame.K_a: "left",
            pygame.K_RIGHT: "right",
            pygame.K_d: "right",
        }
        if key == pygame.K_ESCAPE:
            return False
        if key in key_actions:
            self.auto_play = False
            self._move(key_actions[key], source="manual")
        elif key == pygame.K_r:
            self._reset()
        elif key == pygame.K_v:
            self._start_vlm_request()
        elif key == pygame.K_SPACE:
            if self.vlm is None:
                self.status = "Start with --vlm to enable VLM control"
            else:
                self.auto_play = not self.auto_play
                self.status = (
                    "VLM autoplay enabled" if self.auto_play else "VLM autoplay paused"
                )
        elif key in {
            pygame.K_1,
            pygame.K_2,
            pygame.K_3,
            pygame.K_4,
            pygame.K_5,
            pygame.K_6,
        }:
            names = available_layouts()
            index = key - pygame.K_1
            if index < len(names):
                self._select_layout(names[index])
        return True

    def _select_layout(self, layout_name: str) -> None:
        self.layout_name = layout_name
        self.board = GridSokoban(make_layout(layout_name))
        self.moves = 0
        self.observation_id = 0
        self.previous_observation = None
        self.pending_command = None
        self._previous_turn_committed = False
        self._invalid_response_count = 0
        self._vlm_request_id += 1
        if self.vlm is not None:
            self.vlm.reset()
        self.status = f"Loaded {layout_name} board"

    def _reset(self) -> None:
        self.board.reset()
        self.moves = 0
        self.observation_id = 0
        self.previous_observation = None
        self.pending_command = None
        self._previous_turn_committed = False
        self._invalid_response_count = 0
        self._vlm_request_id += 1
        if self.vlm is not None:
            self.vlm.reset()
        self.status = "Board reset"

    def _move(self, action: str, *, source: str) -> None:
        if self.board.solved:
            self.status = "Solved — press R or 1–6 for another board"
            return
        result = self.board.step(action)
        self.moves += 1
        if result.solved:
            self.auto_play = False
            self.status = f"Solved in {self.moves} moves!"
        elif not result.moved:
            self.status = f"{source.title()} move blocked"
        elif result.pushed:
            self.status = f"{source.title()} pushed a box"
        else:
            self.status = f"{source.title()} moved"

    def _start_vlm_request(self, *, retry_feedback: str | None = None) -> None:
        if self.vlm is None or self.board.solved or self._vlm_in_flight:
            return
        self._draw_board()
        observation = VisualObservation(
            self.observation_id, self.pending_command, _surface_jpeg(self.board_surface)
        )
        if (
            self.previous_observation is not None
            and self.pending_command is not None
            and not self._previous_turn_committed
        ):
            self.vlm.commit(self.previous_observation, self.pending_command)
            self._previous_turn_committed = True
        self._vlm_in_flight = True
        request_id = self._vlm_request_id
        self.status = "VLM is deciding…"
        if retry_feedback is not None:
            self.last_retry_feedback = retry_feedback
        threading.Thread(
            target=self._complete_vlm_request,
            args=(request_id, observation, retry_feedback),
            daemon=True,
        ).start()

    def _complete_vlm_request(
        self,
        request_id: int,
        observation: VisualObservation,
        retry_feedback: str | None,
    ) -> None:
        assert self.vlm is not None
        try:
            response = self.vlm.complete(observation, retry_feedback=retry_feedback)
            result: _VlmResponse | Exception = _VlmResponse(
                output=str(response),
                reasoning=getattr(response, "reasoning", None),
            )
        except Exception as exc:
            result = exc
        self._vlm_results.put((request_id, observation, result))

    def _consume_vlm_result(self) -> None:
        try:
            request_id, observation, result = self._vlm_results.get_nowait()
        except queue.Empty:
            return
        self._vlm_in_flight = False
        if request_id != self._vlm_request_id:
            return
        if isinstance(result, Exception):
            self.auto_play = False
            self.last_vlm_output = f"ERROR: {type(result).__name__}: {result}"
            self.last_vlm_reasoning = "No reasoning: VLM request failed."
            self.status = f"VLM error: {type(result).__name__}: {result}"
            return
        self.last_vlm_output = result.output
        self.last_vlm_reasoning = result.reasoning or "No reasoning field returned."
        try:
            action = parse_action(result.output)
        except ValueError as exc:
            self._invalid_response_count += 1
            if self._invalid_response_count >= MAX_INVALID_VLM_RESPONSES:
                self.auto_play = False
                self.status = f"VLM response error after {MAX_INVALID_VLM_RESPONSES} attempts: {exc}"
                return
            self.status = (
                f"Invalid VLM JSON; retrying "
                f"({self._invalid_response_count + 1}/{MAX_INVALID_VLM_RESPONSES})…"
            )
            self._start_vlm_request(
                retry_feedback=(
                    f"Your previous response was invalid: {exc}. "
                    "Return only the required JSON object."
                )
            )
            return
        self.previous_observation = observation
        self.pending_command = result.output
        self._previous_turn_committed = False
        self._invalid_response_count = 0
        self.observation_id += 1
        self.last_auto_at = time.monotonic()
        self._move(action, source="VLM")

    def _draw(self) -> None:
        self.window.fill(BACKGROUND)
        self._draw_board()
        self.window.blit(self.board_surface, (0, 0))
        panel = pygame.Rect(
            0, self.board.rows * CELL_PIXELS, self.window.get_width(), PANEL_HEIGHT
        )
        pygame.draw.rect(self.window, (34, 41, 53), panel)
        self.window.blit(
            self.font.render(
                f"Grid Sokoban · {self.layout_name} · moves {self.moves}", True, TEXT
            ),
            (16, panel.y + 14),
        )
        controls = "R reset · 1–6 boards · V VLM move · Space autoplay · Esc quit"
        self.window.blit(
            self.small_font.render(controls, True, MUTED_TEXT), (16, panel.y + 43)
        )
        self.window.blit(
            self.small_font.render(self.status, True, TEXT), (16, panel.y + 69)
        )
        self._draw_debug_panel()
        if self.board.solved:
            overlay = pygame.Surface(self.board_surface.get_size(), pygame.SRCALPHA)
            overlay.fill((21, 94, 49, 118))
            self.window.blit(overlay, (0, 0))
            message = self.font.render("SOLVED", True, TEXT)
            self.window.blit(
                message,
                message.get_rect(
                    center=(
                        self.window.get_width() // 2,
                        self.board_surface.get_height() // 2,
                    )
                ),
            )

    def _draw_debug_panel(self) -> None:
        left = self.board.cols * CELL_PIXELS
        panel = pygame.Rect(left, 0, DEBUG_PANEL_WIDTH, self.window.get_height())
        pygame.draw.rect(self.window, (28, 34, 45), panel)
        pygame.draw.line(self.window, (77, 89, 107), (left, 0), (left, panel.height), 2)
        y = 18
        self.window.blit(self.font.render("VLM DEBUG", True, TEXT), (left + 16, y))
        y += 38
        y = self._draw_debug_value(
            "Raw output", self.last_vlm_output, left + 16, y, max_lines=4
        )
        y += 8
        y = self._draw_debug_value(
            "Reasoning", self.last_vlm_reasoning, left + 16, y, max_lines=10
        )
        y += 8
        self._draw_debug_value(
            "Retry feedback", self.last_retry_feedback, left + 16, y, max_lines=4
        )

    def _draw_debug_value(
        self, label: str, value: str, left: int, y: int, *, max_lines: int
    ) -> int:
        self.window.blit(
            self.small_font.render(label, True, (118, 204, 255)), (left, y)
        )
        y += 23
        lines = _wrap_debug_text(value, self.small_font, DEBUG_PANEL_WIDTH - 32)
        for line in lines[:max_lines]:
            self.window.blit(self.small_font.render(line, True, TEXT), (left, y))
            y += 20
        if len(lines) > max_lines:
            self.window.blit(self.small_font.render("…", True, MUTED_TEXT), (left, y))
            y += 20
        return y

    def _draw_board(self) -> None:
        self.board_surface.fill(FLOOR)
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                rect = pygame.Rect(
                    col * CELL_PIXELS, row * CELL_PIXELS, CELL_PIXELS, CELL_PIXELS
                )
                if (row, col) in self.board.walls:
                    pygame.draw.rect(self.board_surface, WALL, rect)
                    pygame.draw.rect(self.board_surface, (91, 99, 111), rect, width=3)
                else:
                    pygame.draw.rect(self.board_surface, GRID, rect, width=1)
        for row, col in self.board.goals:
            rect = pygame.Rect(
                col * CELL_PIXELS,
                row * CELL_PIXELS,
                CELL_PIXELS,
                CELL_PIXELS,
            )
            pygame.draw.rect(self.board_surface, GOAL, rect)
        for row, col in self.board.boxes:
            inset = round(CELL_PIXELS * 0.16)
            rect = pygame.Rect(
                col * CELL_PIXELS + inset,
                row * CELL_PIXELS + inset,
                CELL_PIXELS - 2 * inset,
                CELL_PIXELS - 2 * inset,
            )
            pygame.draw.rect(self.board_surface, BOX, rect)
            pygame.draw.rect(self.board_surface, BOX_EDGE, rect, width=4)
        row, col = self.board.player
        center = ((col + 0.5) * CELL_PIXELS, (row + 0.5) * CELL_PIXELS)
        pygame.draw.circle(
            self.board_surface, PLAYER, center, round(CELL_PIXELS * 0.25)
        )
        pygame.draw.circle(
            self.board_surface,
            (29, 81, 151),
            center,
            round(CELL_PIXELS * 0.25),
            width=3,
        )


def _wrap_debug_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Wrap debug text to the panel width while preserving explicit newlines."""
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split() or [""]:
            candidate = word if not current else f"{current} {word}"
            if current and font.size(candidate)[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def _surface_jpeg(surface: pygame.Surface) -> bytes:
    pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)
    buffer = BytesIO()
    iio.imwrite(buffer, np.ascontiguousarray(pixels), extension=".jpg", quality=90)
    return buffer.getvalue()


def _make_vlm_client() -> OAIChatClient:
    url = os.environ.get("VLM_URL", "").strip().rstrip("/")
    if not url:
        raise ValueError("VLM_URL is required when using --vlm")
    task_dir = Path(__file__).parent
    return OAIChatClient(
        base_url=url,
        timeout=float(os.environ.get("VLM_TIMEOUT", "120")),
        system_prompt=_prompt_path("VLM_SYSTEM_PROMPT", task_dir / "TASK.md").read_text(
            encoding="utf-8"
        ),
        user_prompt=_prompt_path("VLM_USER_PROMPT", task_dir / "USER.md").read_text(
            encoding="utf-8"
        ),
        history_turns=int(os.environ.get("VLM_HISTORY_TURNS", "16")),
        history_retain_turns=int(os.environ.get("VLM_HISTORY_RETAIN_TURNS", "4")),
    )


def _prompt_path(name: str, default: Path) -> Path:
    """Resolve a prompt path supplied by the launcher."""
    return Path(os.environ.get(name, str(default)))


def _env_flag(name: str, *, default: bool) -> bool:
    """Read an optional boolean application setting from the launcher."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive visual 2D Sokoban")
    parser.add_argument(
        "--layout",
        choices=available_layouts(),
        default=os.environ.get("GRID_LAYOUT", "straight"),
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        default=_env_flag("GRID_VLM", default=False),
        help="enable VLM moves with V / Space",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        default=_env_flag("GRID_AUTOPLAY", default=False),
        help="start VLM autoplay",
    )
    args = parser.parse_args()
    if args.autoplay and not args.vlm:
        parser.error("--autoplay requires --vlm")
    SokobanApp(
        layout_name=args.layout,
        vlm=_make_vlm_client() if args.vlm else None,
        auto_play=args.autoplay,
    ).run()


if __name__ == "__main__":
    main()
