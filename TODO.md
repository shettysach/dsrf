# Perf

[ ] ONNX - try TensorrtExecutionProvider vs CUDAExecutionProvider (current) 

[ ] Communication - Replace np with torch / zero-copy tensors

# Context mgmt

[ ] Bound VLM history without defeating llama-server's KV-prefix cache
    - Avoid retaining and base64-encoding every old JPEG indefinitely; epoch rollover must bound client memory, JSON serialization, and HTTP payload size as well as model context usage.
    - The normal server invocation already uses one slot (`--parallel 1`, equivalent to `-np 1`) and prompt caching (`--cache-prompt`).

# Sim

[ ] Per chunk rebasing vs not.

# Structure

[ ] Align kinematic planner commands with NVIDIA inputs: **separate movement/facing directions** 

[ ] Kinematic Planner - Distinguish four-frame `specific_target_positions` from route waypoints.

[ ] Review owner-guaranteed required-field checks in `AgentCommand`, `VisualObservation`, `MotionChunk`, and `PipelineError` for the same simplification.
