#!/usr/bin/env bash

# Record one complete VLM rollout per MP4. Every rollout receives the same
# command budget; it may end earlier only when the VLM explicitly says stand.
set -Eeuo pipefail

VIDEO_DIR="${1:-${VIDEO_DIR:-./videos/sokoban-runs}}"
RUN_COUNT="${RUN_COUNT:-1}"
MAX_COMMANDS="${MAX_COMMANDS:-20}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-600}"
SOKOBAN_LEVELS="${SOKOBAN_LEVELS:-1 2 3 4 5}"

mkdir -p "$VIDEO_DIR"

active_pid=""

finish_current_run() {
    local status=130
    trap - INT TERM
    if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
        echo "Stopping Dora gracefully (waiting for MP4 finalization)..." >&2
        kill -INT "$active_pid" 2>/dev/null || true
        wait "$active_pid" || status=$?
    fi
    exit "$status"
}

trap 'finish_current_run' INT TERM

for level in $SOKOBAN_LEVELS; do
    if ((level < 1 || level > 10)); then
        echo "SOKOBAN_LEVELS accepts levels 1 through 10; got $level" >&2
        exit 2
    fi
    for ((run=1; run<=RUN_COUNT; run++)); do
        output_path="$VIDEO_DIR/sokoban-l$(printf '%02d' "$level")-r$(printf '%02d' "$run").mp4"
        echo "Starting Sokoban level $level, rollout $run/$RUN_COUNT -> $output_path"

        SOKOBAN_LEVEL="$level" DEMO_VIDEO_PATH="$output_path" \
            STOP_ON_STAND=true DEMO_MAX_COMMANDS="$MAX_COMMANDS" \
            DEMO_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS" VIEWER="${VIEWER:-none}" \
            dora run sokoban.yml &
        active_pid=$!
        status=0
        wait "$active_pid" || status=$?
        active_pid=""

        if ((status != 0)); then
            echo "Sokoban level $level rollout $run failed with status $status" >&2
            exit "$status"
        fi
        if [[ ! -s "$output_path" ]]; then
            echo "Run exited without a non-empty video: $output_path" >&2
            exit 1
        fi
        echo "Saved $output_path"
    done
done

echo "Saved Sokoban videos in $VIDEO_DIR"
