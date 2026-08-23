from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from motion_gen.kinematic_planner.generator import KinematicPlanner
from motion_gen.resample import resample_motion
from shared.g1 import DEFAULT_JOINT_POS_MJLAB
from sim.controller.sonic.policy import SonicPolicy
from sim.env import MjlabEnv, RobotState
from sim.renderer import SimRenderer

SONIC_DIR = Path("/tmp/GEAR-SONIC")

pytestmark = pytest.mark.integration
CUDA_READY = torch.cuda.is_available() and "CUDAExecutionProvider" in (
    ort.get_available_providers()
)


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_real_checkpoints_generate_action_and_motion() -> None:
    policy = SonicPolicy(SONIC_DIR)
    encoder_mode = policy.layout.encoder_slices["encoder_mode_4"]
    policy.encoder.input[0, encoder_mode].fill_(1.0)
    state = RobotState(
        root_pos_w=torch.zeros(3),
        root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        root_ang_vel_b=torch.zeros(3),
        projected_gravity_b=torch.tensor([0.0, 0.0, -1.0]),
        joint_pos=torch.as_tensor(DEFAULT_JOINT_POS_MJLAB),
        joint_vel=torch.zeros(29),
    )
    action, completed = policy.infer(state)
    assert not bool(policy.encoder.input[0, encoder_mode].any())
    assert action.shape == (1, 29)
    assert bool(torch.isfinite(action).all())
    assert not completed

    planner = KinematicPlanner(SONIC_DIR / "planner_sonic.onnx")
    planner_qpos = planner.generate("walk", ((1.0, 0.0),))
    chunk = resample_motion(
        planner_qpos,
        source_fps=planner.fps,
        observation_id=0,
        command='{"motion":"walk","waypoints_2d":[[500,500]]}',
    )
    assert 24 <= planner_qpos.shape[0] <= 64
    assert planner_qpos.shape[0] % 4 == 0
    assert chunk.qpos.shape == (planner_qpos.shape[0] * 50 // 30, 36)


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_mjlab_cpu_control_step() -> None:
    simulation = MjlabEnv(device="cpu")
    try:
        policy = SonicPolicy(SONIC_DIR)
        action, _ = policy.infer(simulation.robot_state())
        simulation.step(action)
        assert simulation.unwrapped.common_step_counter == 1
        assert simulation.cfg.sim.njmax == 128
    finally:
        simulation.close()


@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_mjlab_offscreen_capture_is_synchronized_rgbd() -> None:
    simulation = MjlabEnv(
        device="cpu",
        image_width=160,
        image_height=120,
    )
    try:
        jpeg, projection = SimRenderer(simulation, jpeg_quality=80).capture_rgbd()
        assert jpeg.startswith(b"\xff\xd8")
        assert jpeg.endswith(b"\xff\xd9")
        assert projection.depth.shape == (120, 160)
        assert bool(np.isfinite(projection.depth).all())
        assert projection.frustum_height > 0.0
    finally:
        simulation.close()


@pytest.mark.skipif(not CUDA_READY, reason="CUDA Torch and ONNX Runtime are required")
@pytest.mark.skipif(not SONIC_DIR.is_dir(), reason="SONIC bundle is unavailable")
def test_mjlab_and_sonic_share_one_cuda_stream() -> None:
    simulation = MjlabEnv(device="cuda:0")
    try:
        with simulation.compute_context():
            policy = SonicPolicy(
                SONIC_DIR,
                device="cuda:0",
                cuda_stream=simulation.cuda_stream,
            )
            action, _ = policy.infer(simulation.robot_state())

        assert simulation.cuda_stream is not None
        stream_ptr = int(simulation.cuda_stream.cuda_stream)
        assert policy.encoder.cuda_stream_ptr == stream_ptr
        assert policy.decoder.cuda_stream_ptr == stream_ptr
        assert action.device == torch.device("cuda:0")
        simulation.step(action)
        assert simulation.unwrapped.common_step_counter == 1
    finally:
        simulation.close()
