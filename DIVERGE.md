# Demovid divergence from `master`

- `DEMO_VIDEO_PATH` records the VLM observation-camera stream with VLM reasoning, a formatted ARDY command, and waypoint/end-effector markers for the first 0.5 seconds of each decision.
- The box-push observation camera is closer (5 m rather than 6 m).
- End-effector targets are validated against ARDY's final waypoint/root constraint, allowing one command to approach a target and then reach it.
- Recorded push runs stop cleanly only after an explicit VLM terminal `stand` (not an error-recovery fallback); `run_push_videos.sh` saves ten numbered videos and waits for each MP4 to finalize.
- Batch recordings stop after 15 VLM turns by default (including invalid/error retries), or after a 10-minute per-run timeout; both limits are configurable in `run_push_videos.sh`.
