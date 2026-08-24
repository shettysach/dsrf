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
`CONTROLLER: direct` for motion-reference tracking through MJLab's built-in G1
actuator PD model.

Direct tracking uses bounded world-frame virtual springs on the pelvis, torso,
and both ankle-roll bodies. Feet receive position-only forces; pelvis and torso
receive roll/pitch stabilization. It never writes the floating-base pose
directly:

```yaml
CONTROLLER: direct
DIRECT_ROOT_GRAVITY_SUPPORT: "${DIRECT_ROOT_GRAVITY_SUPPORT:-0.3}"
DIRECT_ROOT_Z_KP: "${DIRECT_ROOT_Z_KP:-300}"
DIRECT_ROOT_Z_KD: "${DIRECT_ROOT_Z_KD:-75}"
DIRECT_ROOT_RP_KP: "${DIRECT_ROOT_RP_KP:-75}"
DIRECT_ROOT_RP_KD: "${DIRECT_ROOT_RP_KD:-30}"
DIRECT_FOOT_POS_KP: "${DIRECT_FOOT_POS_KP:-75}"
DIRECT_FOOT_POS_KD: "${DIRECT_FOOT_POS_KD:-15}"
DIRECT_TORSO_RP_KP: "${DIRECT_TORSO_RP_KP:-50}"
DIRECT_TORSO_RP_KD: "${DIRECT_TORSO_RP_KD:-20}"
DIRECT_MAX_FORCE: "${DIRECT_MAX_FORCE:-1000}"
DIRECT_MAX_TORQUE: "${DIRECT_MAX_TORQUE:-200}"
DIRECT_WRENCH_LOG_PATH: "${DIRECT_WRENCH_LOG_PATH:-/tmp/direct-wrench.csv}"
```

Set `DIRECT_WRENCH_LOG_PATH` for a per-frame CSV trace of root reference,
state, errors, applied wrench norms, and clipping flags. Leave it empty to
disable diagnostic logging.

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
