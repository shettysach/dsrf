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
dora run demo.yml
```

- Set `VIEWER: none` in `demo.yml` to disable the window for headless runs.
- Set `REFERENCE_GHOST: "true"` in `demo.yml` to show the active motion
  reference in the native viewer.

## ARDY closed loop

The ARDY graph encodes each command's `motion` field with a local Transformers
model and converts the resolved local waypoint into a root-position constraint.
It generates a five-second reference and carries ARDY's generated
history into the next request:

Set `TEXT_ENCODER_MODEL` and `CHECKPOINTS_DIR` in `ardy.yml`, then run:

```bash
dora run ardy.yml
```

The text model must expose `last_hidden_state` with hidden size 4096. The
encoder applies attention-mask-aware mean pooling and sends a float32 vector to
ARDY.
