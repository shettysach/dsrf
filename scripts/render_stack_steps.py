#!/usr/bin/env python3
"""Save the close isometric view of the stack-steps task."""

import os
from pathlib import Path

from mjlab_scout.config import ScoutConfig
from mjlab_scout.runtime import ScoutRuntime

OUTPUT_PATH = Path(
    os.environ.get("STACK_STEPS_IMAGE_PATH", "/tmp/stack-steps-current-isometric.jpg")
)


def main() -> None:
    runtime = ScoutRuntime(
        ScoutConfig(
            device=os.environ.get("DEVICE", "cuda:0"),
            image_width=int(os.environ.get("IMAGE_WIDTH", "1280")),
            image_height=int(os.environ.get("IMAGE_HEIGHT", "720")),
            preview_seconds=0,
        )
    )
    try:
        runtime.load_task("stack-steps")
        OUTPUT_PATH.write_bytes(runtime.capture_view("overview").image)
    finally:
        runtime.close()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
