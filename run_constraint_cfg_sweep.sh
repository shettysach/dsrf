#!/usr/bin/env bash
# Compare text-only generation with four constraint-guidance strengths.
set -uo pipefail

seed="${ARDY_SEED:-1234}"
log_file="${1:-ardy-cfg-sweep.log}"
failures=0

run() {
  local name="$1"
  local status
  shift
  echo "=== $name (ARDY_SEED=$seed) ===" | tee -a "$log_file"
  ARDY_SEED="$seed" "$@" 2>&1 | tee -a "$log_file"
  status=${PIPESTATUS[0]}
  if ((status != 0)); then
    echo "=== $name FAILED (exit=$status); continuing sweep ===" | tee -a "$log_file"
    failures=1
  fi
}

run text_only dora run arms_forward.yml
for weight in 0 0.5 1.0 2.0; do
  run "constrained_cfg_$weight" env "ARDY_CONSTRAINT_CFG_WEIGHT=$weight" \
    dora run arms_forward_constrained.yml
done

exit "$failures"
