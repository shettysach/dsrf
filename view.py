#!/usr/bin/env python3
"""Save the agent view of every registered task under /tmp."""

from pathlib import Path

from mjlab_scout.config import ScoutConfig
from mjlab_scout.runtime import ScoutRuntime

OUTPUT_DIR = Path("/tmp/dsrf-task-images")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = ScoutRuntime(
        ScoutConfig(
            device="cuda:0",
            image_width=1280,
            image_height=720,
            preview_seconds=0,
        )
    )
    try:
        for task in runtime.list_tasks():
            runtime.load_task(task.name)
            image = runtime.capture_view("agent").image
            path = OUTPUT_DIR / f"{task.name}.jpg"
            path.write_bytes(image)
            print(path)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
