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

## Controllers

The simulator accepts physical G1 joint targets in radians from either controller.
Use `CONTROLLER: sonic` with `SONIC_DIR` for the learned SONIC policy, or
`CONTROLLER: virtual_forces` for direct motion-reference tracking through MJLab's
G1 actuator PD model.

Virtual-force assistance is disabled by default. Enable and tune it with:

```yaml
CONTROLLER: virtual_forces
VF_ASSISTANCE_ENABLED: "true"
VF_TARGET_BODY: pelvis
VF_POSITION_KP: "400"
VF_POSITION_KD: "40"
VF_ORIENTATION_KP: "100"
VF_ORIENTATION_KD: "10"
VF_POSITION_DEADBAND: "0.02"
VF_ORIENTATION_DEADBAND: "0.05"
VF_FORCE_LIMIT: "300"
VF_TORQUE_LIMIT: "100"
```

Forces and torques are world-frame values applied at the named body's center of
mass. Limits are in newtons and newton-metres, respectively.

## ARDY closed loop

The ARDY motion generator encodes each command's `motion` field with a local
Transformers model. It converts resolved floor waypoints into root-position
constraints and visible hand or foot targets into ARDY global-joint-position constraints.
It generates a five-second reference and carries ARDY's generated history into
the next request.

Set `TEXT_ENCODER_MODEL`, `TEXT_ENCODER_DEVICE`, `DEVICE`, and `CHECKPOINTS_DIR`
in `ardy.yml`, then run:

```bash
dora run ardy.yml
```

The text model must expose `last_hidden_state` with hidden size 4096. The
encoder applies attention-mask-aware mean pooling and transfers the resulting
float32 tensor directly to ARDY's device.

## Constraint grounding

The simulator publishes RGB-only observations. When a VLM command contains one
or more image waypoints or end-effector targets, the agent sends those pixels to the
simulator while physics remains paused. The simulator renders depth on demand,
resolves each pixel into a robot-local target, and returns only those coordinates.
The agent then sends one complete request containing the motion prompt and
resolved targets to the motion generator.

Depth is cached for the current observation, so VLM retries and multiple waypoints reuse the same render.
The cache is discarded when motion begins and the next RGB observation is
published.
