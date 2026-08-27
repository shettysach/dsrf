#!/usr/bin/env bash

set -Eeuo pipefail

# Override VIDEO_DIR=/somewhere or pass the directory as the first argument.
VIDEO_DIR="${1:-${VIDEO_DIR:-$HOME/Videos/push-runs}}"
RUN_COUNT="${RUN_COUNT:-10}"
MAX_COMMANDS="${MAX_COMMANDS:-15}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-300}"

mkdir -p "$VIDEO_DIR"

active_pid=""

finish_current_run() {
    local status=0
    trap - INT TERM
    if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
        echo "Stopping Dora gracefully (waiting for MP4 finalization)..." >&2
        kill -INT "$active_pid" 2>/dev/null || true
        wait "$active_pid" || status=$?
    fi
    exit "${status:-130}"
}

trap 'finish_current_run' INT TERM

for ((run=1; run<=RUN_COUNT; run++)); do
    output_path="$VIDEO_DIR/push-$(printf '%02d' "$run").mp4"
    echo "Starting push run $run/$RUN_COUNT -> $output_path"

    DEMO_VIDEO_PATH="$output_path" STOP_ON_STAND=true \
        DEMO_MAX_COMMANDS="$MAX_COMMANDS" \
        DEMO_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS" dora run push.yml &
    active_pid=$!
    status=0
    wait "$active_pid" || status=$?
    active_pid=""

    if ((status != 0)); then
        echo "Push run $run failed with status $status" >&2
        exit "$status"
    fi
    if [[ ! -s "$output_path" ]]; then
        echo "Push run $run exited without a non-empty video: $output_path" >&2
        exit 1
    fi
    echo "Saved $output_path"
done

echo "Saved $RUN_COUNT push videos in $VIDEO_DIR"
