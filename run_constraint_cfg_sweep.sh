#!/usr/bin/env bash
# Compare text-only generation with four constraint-guidance strengths.
set -euo pipefail

seed="${ARDY_SEED:-1234}"
output_dir="${1:-ardy-cfg-sweep}"
mkdir -p "$output_dir"

run() {
  local name="$1"
  shift
  echo "=== $name (ARDY_SEED=$seed) ==="
  ARDY_SEED="$seed" "$@" 2>&1 | tee "$output_dir/$name.log"
}

run text_only dora run arms_forward.yml
for weight in 0 0.5 1.0 2.0; do
  echo "=== constrained_cfg_$weight (ARDY_SEED=$seed) ==="
  ARDY_SEED="$seed" ARDY_CONSTRAINT_CFG_WEIGHT="$weight" \
    dora run arms_forward_constrained.yml 2>&1 \
    | tee "$output_dir/constrained_cfg_$weight.log"
done
