## Setup

dora-rs

```bash
cargo install \
  --git https://github.com/dora-rs/dora.git \
  --tag v1.0.0-rc.4 \
  --locked \
  dora-cli

uv sync --extra cu128
```

SONIC

```bash
uvx --from huggingface_hub hf download nvidia/GEAR-SONIC \
  model_encoder.onnx \
  model_decoder.onnx \
  observation_config.yaml \
  planner_sonic.onnx \
  low_latency/model_encoder.onnx \
  low_latency/model_decoder.onnx \
  low_latency/observation_config.yaml \
  --local-dir /tmp/GEAR-SONIC
```

Also an OpenAI compatible VLM inference server.

## Run

Run OpenAI compatible VLM inference server.

```bash
dora run corridors.yml
# or
dora run sokoban.yml
# or run the ARDY-driven seesaw task
dora run seesaw.yml
# or run the ARDY-driven stairs task
dora run stairs.yml
```

- Set `VIEWER: none` in the selected dataflow to disable the window for headless
  runs.
- Set `REFERENCE_GHOST: "true"` to show the active motion
  reference in the native viewer.
- Set `DEMO_VIDEO_PATH: /tmp/demo.mp4` to record the observation-camera view.
  The MP4 overlays the VLM's reasoning and a formatted ARDY command.

## ARDY closed loop

The ARDY motion generator encodes each command's `motion` field with a local
Transformers model. It converts resolved floor waypoints into root-position
constraints and visible hand or foot targets into ARDY global-joint-position constraints.
The G1 checkpoint generates 52 frames at 25 FPS (2.08 seconds) and carries
ARDY's generated history into the next request.

Set `TEXT_ENCODER_MODEL`, `TEXT_ENCODER_DEVICE`, `DEVICE`, and `CHECKPOINTS_DIR`
in `ardy.yml`, then run:

```bash
dora run ardy.yml
```

`TEXT_ENCODER_MODEL` must identify one of ARDY's supported AeroEx merged
LLM2Vec checkpoints. ARDY performs its own bidirectional LLM2Vec
tokenization, instruction masking, and pooling, then DSRF transfers only the
resulting `[4096]` float32 embedding to ARDY's device.

### Scripted box push

Run the feedback-controlled script without a VLM:

```bash
CHECKPOINTS_DIR=/path/to/checkpoints \
TEXT_ENCODER_MODEL=/path/to/encoder \
DEMO_VIDEO_PATH=/tmp/scripted_push.mp4 \
dora run push_script.yml
```

The robot approaches a box at x=3 m, attempts two-palm contact for up to three
native windows, and pushes toward x=6 m. One script request owns the entire
interaction. Each window uses four actual-state history frames at 25 FPS;
timed intermediate root/hand targets set the pace instead of demanding that
the final goal be reached in 2.08 seconds. Simulation pauses during generation.
Hand targets follow the box's measured pose, but remain motion constraints,
not guaranteed physical contact. There are no welds or virtual-force assistance.

Optional environment overrides:

- `BOX_PUSH_START_X` / `BOX_PUSH_GOAL_X`: box and goal positions (3 / 6 m).
- `PUSH_NAVIGATION_SPEED` / `PUSH_SPEED`: reference pace (0.4 / 0.15 m/s).
- `ARDY_SEED`: diffusion seed (0); `REFERENCE_GHOST`: reference overlay (true).
- `DEMO_VIDEO_PATH`: optional recording path.

Logs report `push_window`, phase transitions, and a final `push_result`.
Success requires the complete box footprint inside the goal, moving slower
than 0.05 m/s for 0.5 seconds. Contact loss triggers at most two reacquisitions;
contact timeout, stalled progress, falls, tipping, sustained tracking error,
or 90 seconds of simulated time terminate with failure. Window exhaustion
alone never reports success. These pacing defaults require physical validation.

This implementation is intentionally limited to the scripted, aligned planar
box push. The VLM tool interface and other one-window commands are unchanged.

## Constraint grounding

The simulator publishes RGB-only observations. When a VLM command contains one
or more image waypoints or end-effector targets, the agent sends those pixels to the
simulator while physics remains paused. The simulator renders depth on demand,
resolves each pixel into a robot-local target, and returns only those coordinates.
The agent then sends one complete request containing the motion prompt and
resolved targets to the simulator process, which generates, resamples, and
tracks the trajectory on its GPU before stepping MJLab.

Depth is cached for the current observation, so VLM retries and multiple waypoints reuse the same render.
The cache is discarded when motion begins and the next RGB observation is
published.
