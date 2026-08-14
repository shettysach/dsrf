from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shared.messages import SONIC_FPS


@dataclass(frozen=True)
class DemoVlmState:
    observation_id: int = -1
    reasoning: str = ""
    command: str = ""


class DemoVideoRecorder:
    """Writes simulation-timed RGB frames with the active VLM decision burned in."""

    def __init__(self, path: Path, *, fps: int = SONIC_FPS) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = iio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        )
        self.frames = 0
        self._closed = False

    def write_frame(self, rgb: np.ndarray, state: DemoVlmState) -> None:
        self._writer.append_data(compose_demo_frame(rgb, state))
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        self._closed = True


def compose_demo_frame(rgb: np.ndarray, state: DemoVlmState) -> np.ndarray:
    """Burn a compact, resolution-aware VLM panel into one RGB frame."""

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    shortest_side = min(image.size)
    font_size = max(12, min(18, round(shortest_side / 40)))
    font, bold_font = _load_fonts(font_size)
    line_height = font_size + max(4, round(font_size * 0.28))
    padding = max(9, round(font_size * 0.75))
    margin = max(7, round(shortest_side * 0.015))
    panel_width = min(
        image.width - 2 * margin,
        max(160, min(380, round(image.width * 0.28))),
    )
    text_width = panel_width - 2 * padding

    # Four lines are reserved for labels and the command. Reasoning receives only
    # the remaining vertical budget, so it can never push the panel off-frame.
    max_panel_height = round(image.height * 0.55)
    max_entry_lines = max(5, (max_panel_height - 2 * padding - 4) // line_height)
    max_reasoning_lines = max(1, max_entry_lines - 4)
    reasoning = " ".join((state.reasoning.strip() or "No reasoning returned.").split())
    reasoning_lines = _wrap_pixels(
        draw,
        reasoning,
        font,
        max_width=text_width,
        max_lines=max_reasoning_lines,
    )
    entries = [
        (f"Observation #{state.observation_id}", font),
        ("Reasoning", bold_font),
        *[(line, font) for line in reasoning_lines],
        ("Command", bold_font),
        (_decision_label(state.command), font),
    ]

    panel_height = 2 * padding + line_height * len(entries) + 4
    panel_x = image.width - panel_width - margin
    panel_y = image.height - panel_height - margin
    draw.rounded_rectangle(
        (
            panel_x,
            panel_y,
            panel_x + panel_width - 1,
            panel_y + panel_height - 1,
        ),
        radius=max(5, round(font_size * 0.4)),
        fill=(0, 0, 0, 190),
    )
    y = panel_y + padding
    for line, entry_font in entries:
        draw.text(
            (panel_x + padding, y),
            line,
            fill=(255, 255, 255, 255),
            font=entry_font,
        )
        y += line_height

    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


@lru_cache(maxsize=8)
def _load_fonts(
    size: int,
) -> tuple[
    ImageFont.ImageFont | ImageFont.FreeTypeFont,
    ImageFont.ImageFont | ImageFont.FreeTypeFont,
]:
    font = ImageFont.load_default(size=size)
    try:
        bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        bold = font
    return font, bold


def _wrap_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    *,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = text.split() or [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    if len(lines) <= max_lines:
        return lines

    visible = lines[:max_lines]
    visible[-1] = _fit_ellipsis(draw, visible[-1], font, max_width)
    return visible


def _fit_ellipsis(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    suffix = "…"
    clipped = text.rstrip()
    while clipped and draw.textlength(clipped + suffix, font=font) > max_width:
        clipped = clipped[:-1].rstrip()
    return clipped + suffix


def _decision_label(command: str) -> str:
    try:
        payload = json.loads(command)
    except (json.JSONDecodeError, TypeError):
        return command or "WAIT"
    if not isinstance(payload, dict):
        return command or "WAIT"
    motion = payload.get("motion")
    direction = payload.get("direction")
    if isinstance(motion, str) and isinstance(direction, str):
        return f"{motion} {direction}".upper()
    if isinstance(motion, str):
        return motion.upper()
    return command or "WAIT"
