#!/usr/bin/env python3
"""Record a short, overlay-free GIF of the see-saw settling under gravity."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as iio
import torch
from tasks import get_task

from shared.messages import SONIC_FPS
from sim.env import MjlabEnv


def main() -> None:
    output_path = Path(
        os.environ.get("SEESAW_GIF_PATH", "/tmp/see-saw.gif")
    ).expanduser()
    duration = _positive_float("SEESAW_GIF_SECONDS", 3.0)
    fps = _positive_int("SEESAW_GIF_FPS", 20)
    width = _positive_int("IMAGE_WIDTH", 640)
    height = _positive_int("IMAGE_HEIGHT", 480)
    device = os.environ.get("DEVICE", "cuda:0")

    simulation = MjlabEnv(
        device=device,
        image_width=width,
        image_height=height,
        task=get_task("see-saw"),
    )
    frames = []
    try:
        with simulation.compute_context():
            robot = simulation._env.scene["robot"]
            frozen_root = robot.data.default_root_state.clone()
            frozen_joint_pos = robot.data.default_joint_pos.clone()
            frozen_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
        action = torch.zeros(
            (simulation.num_envs, frozen_joint_pos.shape[1]),
            dtype=torch.float32,
            device=simulation.device,
        )
        frame_count = max(2, round(duration * fps))
        simulated_steps = 0
        for frame_index in range(frame_count):
            target_step = round(frame_index * SONIC_FPS / fps)
            while simulated_steps < target_step:
                simulation.step(action)
                with simulation.compute_context():
                    robot.write_root_state_to_sim(frozen_root)
                    robot.write_joint_state_to_sim(
                        frozen_joint_pos, frozen_joint_vel
                    )
                    simulation._env.sim.forward()
                simulated_steps += 1
            frames.append(simulation.render_demo_rgb())
    finally:
        simulation.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(
        output_path,
        frames,
        format="GIF",
        duration=1000.0 / fps,
        loop=0,
    )
    print(f"Saved {len(frames)} frames to {output_path}")


def _positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


if __name__ == "__main__":
    main()
