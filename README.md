# DSRF contact-free push motion

This repository's runnable dataflow is `push_motion_script.yml`. A script sends
one root-path goal to ARDY; the simulator generates 52 frames at 25 FPS per
window and tracks them with SONIC. There is no VLM, box, contact requirement,
or virtual-force assistance in this flow. Other task implementations remain in
the source tree but have no maintained launchers.

## Setup

Install [dora](https://github.com/dora-rs/dora), then install project dependencies:

```bash
uv sync --extra cu128 --extra ardy
```

Download the licensed `nvidia/ARDY-G1-RP-25FPS-Horizon52` checkpoint into a
directory and set `CHECKPOINTS_DIR` to its parent. Supply a supported ARDY
AeroEx merged LLM2Vec text encoder via `TEXT_ENCODER_MODEL`. SONIC files must
be present at `/tmp/GEAR-SONIC`, or change `SONIC_DIR` in the dataflow.
A CUDA-capable machine is required for the configured dataflow.

## Run

```bash
CHECKPOINTS_DIR=/path/to/checkpoints \
TEXT_ENCODER_MODEL=/path/to/encoder \
dora run push_motion_script.yml
```

`DEMO_VIDEO_PATH=/tmp/push_motion.mp4` records a video. Set `PUSH_MOTION_HANDS=false`
to diagnose root motion alone; the default constrains both hand positions.
`PUSH_MOTION_APPROACH_X`, `PUSH_MOTION_GOAL_X`, `PUSH_MOTION_NAVIGATION_SPEED`,
and `PUSH_MOTION_SPEED` tune the physical path and pacing.

The script walks to x=2 m, reaches, then walks with both hands forward to the
green line at x=6 m. Per-window root and hand keyframes maintain the pace;
phase deadlines remain at absolute script-frame times and enter ARDY's
10-second conditioning horizon when visible. Every window is replanned from
measured robot state. Success requires the measured root within 0.10 m of the
goal, height at least 0.55 m, and both measured hands at least 0.20 m ahead
of the root, before the 90-second simulation timeout. These thresholds and
pace defaults require validation on the target GPU and tracker.
