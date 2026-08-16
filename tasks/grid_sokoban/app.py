from __future__ import annotations

import argparse
import os
import time
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
FPS = 60
AUTO_DELAY_SECONDS = 0.2
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
        self.last_auto_at = time.monotonic()
        pygame.init()
        pygame.display.set_caption("Grid Sokoban")
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        window_size = (
            self.board.cols * CELL_PIXELS,
            self.board.rows * CELL_PIXELS + PANEL_HEIGHT,
        )
        self.window = pygame.display.set_mode(window_size)
        self.board_surface = pygame.Surface(
            (window_size[0], window_size[1] - PANEL_HEIGHT)
        )

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
            if (
                self.auto_play
                and not self.board.solved
                and time.monotonic() - self.last_auto_at >= AUTO_DELAY_SECONDS
            ):
                self._ask_vlm()
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
            self._ask_vlm()
        elif key == pygame.K_SPACE:
            if self.vlm is None:
                self.status = "Start with --vlm to enable VLM control"
            else:
                self.auto_play = not self.auto_play
                self.status = (
                    "VLM autoplay enabled" if self.auto_play else "VLM autoplay paused"
                )
        elif key in {pygame.K_1, pygame.K_2, pygame.K_3}:
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
        if self.vlm is not None:
            self.vlm.reset()
        self.status = f"Loaded {layout_name} board"

    def _reset(self) -> None:
        self.board.reset()
        self.moves = 0
        self.observation_id = 0
        self.previous_observation = None
        self.pending_command = None
        if self.vlm is not None:
            self.vlm.reset()
        self.status = "Board reset"

    def _move(self, action: str, *, source: str) -> None:
        if self.board.solved:
            self.status = "Solved — press R or 1/2/3 for another board"
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

    def _ask_vlm(self) -> None:
        if self.vlm is None or self.board.solved:
            return
        self._draw_board()
        observation = VisualObservation(
            self.observation_id, self.pending_command, _surface_jpeg(self.board_surface)
        )
        if self.previous_observation is not None and self.pending_command is not None:
            self.vlm.commit(self.previous_observation, self.pending_command)
        try:
            response = self.vlm.complete(observation)
            action = parse_action(str(response))
        except Exception as exc:
            self.auto_play = False
            self.status = f"VLM error: {type(exc).__name__}: {exc}"
            return
        self.previous_observation = observation
        self.pending_command = str(response).strip()
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
        controls = "R reset · 1/2/3 boards · V VLM move · Space autoplay · Esc quit"
        self.window.blit(
            self.small_font.render(controls, True, MUTED_TEXT), (16, panel.y + 43)
        )
        self.window.blit(
            self.small_font.render(self.status, True, TEXT), (16, panel.y + 69)
        )
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
            center = ((col + 0.5) * CELL_PIXELS, (row + 0.5) * CELL_PIXELS)
            pygame.draw.circle(
                self.board_surface, GOAL, center, round(CELL_PIXELS * 0.27)
            )
            pygame.draw.circle(
                self.board_surface,
                (38, 119, 65),
                center,
                round(CELL_PIXELS * 0.27),
                width=3,
            )
        for row, col in self.board.boxes:
            inset = round(CELL_PIXELS * 0.16)
            rect = pygame.Rect(
                col * CELL_PIXELS + inset,
                row * CELL_PIXELS + inset,
                CELL_PIXELS - 2 * inset,
                CELL_PIXELS - 2 * inset,
            )
            pygame.draw.rect(self.board_surface, BOX, rect, border_radius=8)
            pygame.draw.rect(
                self.board_surface, BOX_EDGE, rect, width=4, border_radius=8
            )
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


def _surface_jpeg(surface: pygame.Surface) -> bytes:
    pixels = pygame.surfarray.array3d(surface).swapaxes(0, 1)
    buffer = BytesIO()
    iio.imwrite(buffer, np.ascontiguousarray(pixels), extension=".jpg", quality=90)
    return buffer.getvalue()


def _make_vlm_client() -> OAIChatClient:
    url = os.environ.get("VLM_URL", "").strip().rstrip("/")
    if not url:
        raise ValueError("VLM_URL is required when using --vlm")
    return OAIChatClient(
        base_url=url,
        timeout=float(os.environ.get("VLM_TIMEOUT", "120")),
        system_prompt=Path(__file__).with_name("SYSTEM.md").read_text(encoding="utf-8"),
        user_prompt=(
            Path(__file__).parents[2] / "prompt" / "GRID_SOKOBAN_USER.md"
        ).read_text(encoding="utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive visual 2D Sokoban")
    parser.add_argument("--layout", choices=available_layouts(), default="straight")
    parser.add_argument(
        "--vlm", action="store_true", help="enable VLM moves with V / Space"
    )
    parser.add_argument("--autoplay", action="store_true", help="start VLM autoplay")
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
