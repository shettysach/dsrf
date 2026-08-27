"""Simple MP4 recording with the active VLM decision overlaid on each frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shared.messages import REFERENCE_HZ

type Font = ImageFont.ImageFont | ImageFont.FreeTypeFont


@dataclass(frozen=True)
class DemoVlmState:
    observation_id: int = -1
    reasoning: str = ""
    command: str = ""


class DemoVideoRecorder:
    def __init__(self, path: Path, *, fps: int = REFERENCE_HZ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = iio.get_writer(
            str(path), fps=fps, codec="libx264", macro_block_size=1
        )
        self._closed = False
        self._marked_observation_id: int | None = None

    def write_frame(self, rgb: np.ndarray, state: DemoVlmState) -> None:
        show_targets = state.observation_id != self._marked_observation_id
        self._writer.append_data(
            compose_demo_frame(rgb, state, show_targets=show_targets)
        )
        if show_targets:
            self._marked_observation_id = state.observation_id

    def close(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True


def compose_demo_frame(
    rgb: np.ndarray, state: DemoVlmState, *, show_targets: bool = True
) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(14, min(22, min(image.size) // 32))
    font = ImageFont.load_default(size=font_size)
    bold = _bold_font(font_size, font)
    padding = max(10, font_size // 2)
    panel_width = min(image.width - 20, max(300, image.width // 2))
    text_width = panel_width - 2 * padding
    lines = [
        (f"VLM decision · observation {state.observation_id}", bold),
        ("Reasoning", bold),
        *[
            (line, font)
            for line in _wrap(
                draw, state.reasoning or "(not returned)", font, text_width, 5
            )
        ],
        ("ARDY command", bold),
        *[
            (line, font)
            for line in _wrap(draw, _format_ardy(state.command), font, text_width, 4)
        ],
    ]
    line_height = font_size + 5
    panel_height = padding * 2 + line_height * len(lines)
    x, y = 10, 10
    draw.rounded_rectangle(
        (x, y, x + panel_width, y + panel_height), radius=8, fill=(0, 0, 0, 185)
    )
    cursor_y = y + padding
    for text, text_font in lines:
        draw.text((x + padding, cursor_y), text, fill="white", font=text_font)
        cursor_y += line_height
    if show_targets:
        _draw_vlm_targets(draw, state.command, image.size, font)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _bold_font(size: int, fallback: Font) -> Font:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return fallback


def _wrap(
    draw: ImageDraw.ImageDraw, text: str, font: Font, width: int, limit: int
) -> list[str]:
    words = " ".join(text.split()).split() or [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1].rstrip(".") + "…"
    return lines


def _format_ardy(command: str) -> str:
    try:
        payload = json.loads(command)
    except (json.JSONDecodeError, TypeError):
        return command or "(no command)"
    if not isinstance(payload, dict):
        return command
    fields = [f"motion: {payload.get('motion', 'unknown')}"]
    if payload.get("waypoints_2d"):
        fields.append(f"waypoints: {payload['waypoints_2d']}")
    if payload.get("end_effectors"):
        fields.append(f"end effectors: {payload['end_effectors']}")
    return " · ".join(fields)


def _draw_vlm_targets(
    draw: ImageDraw.ImageDraw,
    command: str,
    image_size: tuple[int, int],
    font: Font,
) -> None:
    """Draw VLM-provided normalized image targets over the recorded image."""
    try:
        payload = json.loads(command)
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(payload, dict):
        return

    for point in payload.get("waypoints_2d", []):
        normalized = _normalized_point(point)
        if normalized is not None:
            _draw_marker(draw, normalized, image_size, color=(35, 120, 255), font=font)
    for end_effector in payload.get("end_effectors", []):
        if not isinstance(end_effector, dict):
            continue
        normalized = _normalized_point(end_effector.get("target_2d"))
        name = end_effector.get("name")
        if normalized is not None and isinstance(name, str):
            _draw_marker(
                draw,
                normalized,
                image_size,
                color=(235, 30, 30),
                label=name,
                font=font,
            )


def _normalized_point(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    x, y = value
    if any(
        isinstance(coordinate, bool) or not isinstance(coordinate, int)
        for coordinate in (x, y)
    ):
        return None
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    image_size: tuple[int, int],
    *,
    color: tuple[int, int, int],
    font: Font,
    label: str | None = None,
) -> None:
    width, height = image_size
    x = round(point[0] / 1000 * (width - 1))
    y = round(point[1] / 1000 * (height - 1))
    radius = max(5, min(width, height) // 50)
    draw.ellipse(
        (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill="white"
    )
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    if label is not None:
        label_x, label_y = x + radius + 4, y - radius
        bbox = draw.textbbox((label_x, label_y), label, font=font)
        draw.rounded_rectangle(
            (bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2),
            radius=3,
            fill=(0, 0, 0),
        )
        draw.text((label_x, label_y), label, fill="white", font=font)
