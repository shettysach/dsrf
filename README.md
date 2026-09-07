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

### Sokoban recordings

Use one video for one full rollout: it preserves the complete sequence of VLM
decisions and makes failures directly comparable. The batch launcher records
each selected level separately, with the same maximum number of VLM commands.

```bash
# All ten levels, one rollout each; lazy Viser viewer; 20 VLM commands maximum
# per rollout. Use VIEWER=none for no viewer server at all.
./run_sokoban_videos.sh

# The optional Viser scene viewer is at http://localhost:8081 by default.
# Override its port with VISER_PORT when needed.

# Three independent attempts on levels 1 and 5, with a 30-command budget.
SOKOBAN_LEVELS="1 5" RUN_COUNT=3 MAX_COMMANDS=30 ./run_sokoban_videos.sh /tmp/sokoban-videos
```

Each clip stops as soon as the VLM issues `stand`; otherwise its final command
is marked terminal at `MAX_COMMANDS`. `RUN_TIMEOUT_SECONDS` (600 by default) is
only a safety stop. For a single interactive run, use
`SOKOBAN_LEVEL=4 DEMO_VIDEO_PATH=/tmp/sokoban.mp4 STOP_ON_STAND=true DEMO_MAX_COMMANDS=20 dora run sokoban.yml`.

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

For the direct physical-contact baseline, run the same script with its welds
disabled:

```bash
PUSH_WELD=false dora run push_script.yml
```

The robot approaches a box at x=3 m, reaches for two native windows, and pushes
toward x=6 m. One script request owns the entire
interaction. Each window uses four actual-state history frames at 25 FPS;
Timed intermediate root/hand targets set the pace instead of demanding that
the final goal be reached in 2.08 seconds. Simulation pauses during generation.
ARDY receives only concise phase prompts such as `walk forward`, `reach forward
with both hands`, or `stand`. Hand targets follow the box's measured pose. The
script spends two native windows reaching. By default, contact-gated virtual
force assists the box only while a palm physically touches it. Setting
`PUSH_WELD=true` instead captures current hand-to-box poses in two predeclared,
initially inactive MuJoCo welds and enables them for the push; welds and virtual
force are intentionally mutually exclusive. The welds detach before free-box
settling and always detach on cleanup.

All box-push defaults—including geometry, palm targets, phase prompts, pacing,
and assistance—live in `tasks/box_push/settings.py`. Optional environment
overrides are limited to:

- `BOX_PUSH_START_X` / `BOX_PUSH_GOAL_X`: box and goal positions (3 / 6 m).
- `PUSH_NAVIGATION_SPEED` / `PUSH_SPEED`: reference pace (0.4 / 0.15 m/s).
- `PUSH_STANDOFF`: base-to-contact staging distance (0.35 m).
- `PUSH_CONTACT_WINDOWS`: number of native reach windows (2).
- `PUSH_WELD`: enable scripted hand-to-box welds (`false` by default). It cannot
  be combined with the task's virtual-force assistance.
- `ARDY_SEED`: diffusion seed (0); `REFERENCE_GHOST`: reference overlay (true).
- `DEMO_VIDEO_PATH`: optional recording path.

Logs report `push_window`, phase transitions, and a final `push_result`.
Success requires the complete box footprint inside the goal, moving slower
than 0.05 m/s for 0.5 seconds. Stalled progress or 90 seconds of simulated
time terminate with failure. Window exhaustion alone never reports success.
These pacing defaults require physical validation.

This implementation is intentionally limited to the scripted, aligned planar
box push. The VLM tool interface and other one-window commands are unchanged.

### Contact-free push motion

For the same bilateral push gesture without a box, contact requirement, weld,
or virtual-force assistance, run:

```bash
CHECKPOINTS_DIR=/path/to/checkpoints \
TEXT_ENCODER_MODEL=/path/to/encoder \
dora run push_motion_script.yml
```

This uses timed root targets for x=2 m and the green line at x=6 m, plus sparse
bilateral hand keyframes with forward-facing palm normals. There are no torso,
upright-root, or contact constraints. Change the root targets with
`PUSH_MOTION_APPROACH_X` and `PUSH_MOTION_GOAL_X`.

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
