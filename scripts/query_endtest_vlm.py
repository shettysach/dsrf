#!/usr/bin/env python3
"""Ask the configured VLM to select an endtest foot target and mark its pixel."""

import os
from io import BytesIO
from pathlib import Path

import imageio.v3 as iio
from render_endtest import _draw_target_dots

from agent.ardy import ARDY_TOOL
from agent.vlm import OAIChatClient
from motion_gen.ardy.parser import parse_ardy_command
from shared.messages import VisualObservation

IMAGE_PATH = Path("artifacts/endtest-overhead.jpg")
OUTPUT_PATH = Path("artifacts/endtest-vlm-selection.png")


def main() -> None:
    image = IMAGE_PATH.read_bytes()
    client = OAIChatClient(
        base_url=os.environ.get("VLM_URL", "http://127.0.0.1:8080"),
        timeout=float(os.environ.get("VLM_TIMEOUT", "120")),
        system_prompt=Path("tasks/endtest/TASK.md").read_text(encoding="utf-8"),
        user_prompt=Path("prompt/USER.md").read_text(encoding="utf-8"),
        tool=ARDY_TOOL,
    )
    completion = client.complete(VisualObservation(0, None, image))
    command = parse_ardy_command(completion.command)
    if not command.end_effectors:
        raise RuntimeError("The VLM did not select an end-effector target")

    frame = iio.imread(BytesIO(image), extension=".jpg")
    height, width = frame.shape[:2]
    centers = [
        (
            round(selection.target_2d[1] / 1000 * (height - 1)),
            round(selection.target_2d[0] / 1000 * (width - 1)),
        )
        for selection in command.end_effectors
    ]
    iio.imwrite(OUTPUT_PATH, _draw_target_dots(frame, centers), extension=".png")
    print(completion.command)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
