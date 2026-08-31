#!/usr/bin/env bash
# Compare text-only, text-plus-constraint, and constraint-only right kicks.
set -uo pipefail

seed="${ARDY_SEED:-1234}"
text_weight="${ARDY_TEXT_CFG_WEIGHT:-2.0}"
constraint_weight="${ARDY_CONSTRAINT_CFG_WEIGHT:-2.0}"
log_file="${1:-right-kick-cfg-sweep.log}"
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

run text_only env ARDY_TEXT_CFG_WEIGHT="$text_weight" dora run right_kick.yml
run text_plus_constraint env ARDY_TEXT_CFG_WEIGHT="$text_weight" \
  ARDY_CONSTRAINT_CFG_WEIGHT="$constraint_weight" dora run right_kick_constrained.yml
run constraint_only env ARDY_TEXT_CFG_WEIGHT=0.0 \
  ARDY_CONSTRAINT_CFG_WEIGHT="$constraint_weight" dora run right_kick_constraint_only.yml

exit "$failures"
