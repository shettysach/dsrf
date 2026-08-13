[ ] ONNX - try TensorrtExecutionProvider vs CUDAExecutionProvider (current) 

[ ] Bound VLM history without defeating llama-server's KV-prefix cache
    - Avoid retaining and base64-encoding every old JPEG indefinitely; epoch rollover must bound client memory, JSON serialization, and HTTP payload size as well as model context usage.
    - The normal server invocation already uses one slot (`--parallel 1`, equivalent to `-np 1`) and prompt caching (`--cache-prompt`).

[ ] Per chunk rebasing vs not.

[ ] Integrate text_encoder node functionality into motion_gen

[ ] ALT_PATH - camera details could be sent via agent node than directly to the motion_gen node. `n_envs` cost goes high.

[ ] ARDY waypoint convention: `target_xy` is robot-local `(forward, left)`, but
    `build_waypoint_constraints()` currently only swaps it to ARDY `(left, forward)`
    and adds it in ARDY's fixed root frame. Rotate the local target by the current
    root heading before applying it, so waypoints remain correct after turns.

[ ] Collision detection, hooks.
