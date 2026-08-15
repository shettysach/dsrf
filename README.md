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

Also a configured [`pi`](https://pi.dev) CLI. The agent launches Pi in RPC mode;
configure Pi's model provider, model, extensions, session storage, and compaction before
starting Dora. The dataflow supplies only the task-specific system and command prompts.
Set `PI_DEBUG=1` in the agent environment to print a compact live trace of Pi's assistant
text, tool calls, retries, errors, and settled events. Image payloads and raw RPC JSON are
never printed.

## Run

```bash
dora run corridors.yml
# or
dora run sokoban.yml
```

- Set `VIEWER: none` in the selected dataflow to disable the window for headless
  runs.
- Set `REFERENCE_GHOST: "true"` to show the active motion
  reference in the native viewer.

## ARDY closed loop

The ARDY motion generator encodes each command's `motion` field with a local
Transformers model and converts the resolved local waypoint into a root-position
constraint. It generates a five-second reference and carries ARDY's generated
history into the next request.

Set `TEXT_ENCODER_MODEL`, `TEXT_ENCODER_DEVICE`, `DEVICE`, and `CHECKPOINTS_DIR`
in `ardy.yml`, then run:

```bash
dora run ardy.yml
```

The text model must expose `last_hidden_state` with hidden size 4096. The
encoder applies attention-mask-aware mean pooling and transfers the resulting
float32 tensor directly to ARDY's device.

## Waypoint grounding

The simulator publishes RGB-only observations. When a VLM command contains an
one or more image waypoints, the agent sends those pixels to the simulator while physics remains
paused. The simulator renders depth on demand, resolves the pixel into a
robot-local targets, and returns only those coordinates. The agent then
sends one complete request containing the motion prompt and ordered resolved targets to
the motion generator.

Depth is cached for the current observation, so VLM retries and multiple waypoints reuse the same render.
The cache is discarded when motion begins and the next RGB observation is
published.
