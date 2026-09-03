from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.skeleton import G1Skeleton34

from motion_gen.ardy.encoder import prepare_conditioning
from motion_gen.ardy.history import qpos_to_ardy_inputs
from shared.g1 import standing_qpos
from shared.messages import EndEffectorTarget


def test_ardy_model_loader_receives_a_device_string(monkeypatch, tmp_path) -> None:
    import motion_gen.ardy.generator as ardy_generator

    received: dict[str, object] = {}
    model = SimpleNamespace(
        motion_rep=SimpleNamespace(fps=25),
        skeleton=object(),
        num_frames_per_token=4,
        gen_horizon_len=52,
    )
    monkeypatch.setattr(
        ardy_generator,
        "load_model",
        lambda *args, **kwargs: received.update(kwargs) or model,
    )
    monkeypatch.setattr(ardy_generator, "MujocoQposConverter", lambda _: object())
    monkeypatch.setattr(
        ardy_generator,
        "qpos_to_ardy_inputs",
        lambda *args, **kwargs: (
            torch.zeros((1, 4, 1, 3, 3)),
            torch.zeros((1, 4, 3)),
        ),
    )
    generator = ardy_generator.Ardy(tmp_path, device="cuda:0")

    assert received["device"] == "cuda:0"
    assert generator.history_crop_frames == 4
    assert generator.motion_history is None


def test_standing_qpos_round_trips_through_ardy_inputs() -> None:
    converter = MujocoQposConverter(G1Skeleton34())
    expected = np.repeat(standing_qpos()[None], 4, axis=0)

    local_rot_mats, root_positions = qpos_to_ardy_inputs(
        expected,
        converter,
        device=torch.device("cpu"),
    )
    restored = converter.dict_to_qpos(
        {
            "local_rot_mats": local_rot_mats,
            "root_positions": root_positions,
        },
        "cpu",
    )

    np.testing.assert_allclose(restored[0], expected, atol=1e-5)


def test_ardy_history_conditions_continuation_and_is_retained() -> None:
    import motion_gen.ardy.generator as ardy_generator

    model = Mock()
    model.gen_horizon_len = 52
    model.diffusion.num_base_steps = 10
    returned_motion = torch.arange(56 * 3, dtype=torch.float32).reshape(1, 56, 3)
    model.autoregressive_step.return_value = returned_motion
    model.motion_rep.create_conditions.return_value = (
        torch.zeros((56, 3)),
        torch.zeros((56, 3)),
    )
    model.motion_rep.inverse.side_effect = lambda motion, **kwargs: {
        "motion": motion,
        "root_positions": torch.zeros((1, 52, 3)),
        "global_root_heading": torch.tensor([[[1.0, 0.0]] * 51 + [[0.0, 1.0]]]),
    }
    converter = Mock()
    converter.dict_to_qpos.return_value = torch.zeros((1, 52, 36))

    generator = ardy_generator.Ardy.__new__(ardy_generator.Ardy)
    generator.device = torch.device("cpu")
    generator.model = model
    generator.converter = converter
    generator.history_crop_frames = 4
    generator.motion_history = torch.zeros((1, 4, 3))
    generator.root_history = torch.zeros((2, 3))
    generator.root_heading = torch.tensor(0.0)
    initial_history = generator.motion_history

    embedding = torch.arange(4096, dtype=torch.float32)
    qpos = generator.generate(embedding, ((0.0, 0.5),))

    assert qpos.shape == (52, 36)
    assert converter.dict_to_qpos.call_args.kwargs["numpy"] is False
    model.motion_rep.inverse.assert_called_once()
    generated_motion = model.motion_rep.inverse.call_args.args[0]
    assert generated_motion.shape == (1, 52, 3)
    call = model.autoregressive_step.call_args
    assert call.kwargs["num_frames"] == 56
    assert call.kwargs["init_history_sequence"] is initial_history
    assert call.kwargs["motion_mask"] is not None
    assert call.kwargs["observed_motion"] is not None
    assert call.kwargs["text_feat"].shape == (1, 1, 4096)
    torch.testing.assert_close(call.kwargs["text_feat"][0, 0], embedding)
    torch.testing.assert_close(generator.motion_history, returned_motion[:, -4:])
    torch.testing.assert_close(generator.root_history, torch.zeros((2, 3)))
    torch.testing.assert_close(generator.root_heading, torch.tensor(torch.pi / 2.0))


def test_first_generation_uses_initial_pose_then_retains_continuation_window() -> None:
    import motion_gen.ardy.generator as ardy_generator

    model = Mock()
    model.gen_horizon_len = 52
    model.diffusion.num_base_steps = 10
    first_motion = torch.zeros((1, 52, 3))
    second_motion = torch.ones((1, 56, 3))
    model.autoregressive_step.side_effect = [first_motion, second_motion]
    model.motion_rep.inverse.return_value = {
        "root_positions": torch.zeros((1, 52, 3)),
        "global_root_heading": torch.tensor([[[1.0, 0.0]] * 52]),
    }
    converter = Mock()
    converter.dict_to_qpos.return_value = torch.zeros((1, 52, 36))

    generator = ardy_generator.Ardy.__new__(ardy_generator.Ardy)
    generator.device = torch.device("cpu")
    generator.model = model
    generator.converter = converter
    generator.history_crop_frames = 4
    generator.motion_history = None
    generator.root_history = torch.tensor([[1.0, 0.8, 3.0], [4.0, 0.8, 6.0]])
    generator.root_heading = torch.tensor(0.0)
    generator.generate(torch.arange(4096, dtype=torch.float32), ())
    first_history = first_motion[:, -4:]
    generator.generate(torch.arange(4096, dtype=torch.float32), ())

    assert model.motion_rep.create_conditions.call_count == 0
    assert model.autoregressive_step.call_count == 2
    first_call, second_call = model.autoregressive_step.call_args_list
    assert first_call.kwargs["num_frames"] == 52
    assert "init_history_sequence" not in first_call.kwargs
    torch.testing.assert_close(
        first_call.kwargs["init_global_translation"],
        torch.tensor([[4.0, 0.0, 6.0]]),
    )
    torch.testing.assert_close(
        first_call.kwargs["init_first_heading_angle"], torch.tensor([0.0])
    )
    assert second_call.kwargs["num_frames"] == 56
    torch.testing.assert_close(second_call.kwargs["init_history_sequence"], first_history)
    assert second_call.kwargs["cfg_weight"] == 2.0
    torch.testing.assert_close(generator.motion_history, second_motion[:, -4:])


def test_ee_condition_comparison_changes_only_the_condition_tensors(
    monkeypatch,
) -> None:
    import motion_gen.ardy.diagnostics as diagnostics

    model = Mock()
    model.gen_horizon_len = 52
    model.diffusion.num_base_steps = 10
    first_motion = torch.zeros((1, 52, 3))
    second_motion = torch.ones((1, 52, 3))
    native_motion = torch.full((1, 52, 3), 3.0)
    model.autoregressive_step.side_effect = [first_motion, second_motion, native_motion]
    model.motion_rep.inverse.side_effect = [
        {
            "posed_joints": torch.zeros((1, 52, 3, 3)),
            "global_rot_mats": torch.eye(3).expand(1, 52, 3, 3, 3),
        },
        {"posed_joints": torch.ones((1, 52, 3, 3))},
        {"posed_joints": torch.full((1, 52, 3, 3), 3.0)},
    ]
    model.motion_rep.create_conditions_from_constraints.return_value = (
        torch.full((52, 3), 4.0),
        torch.ones((52, 3)),
    )
    model.motion_rep.normalize.side_effect = lambda value: value
    motion_mask = torch.ones((1, 52, 3))
    observed_motion = torch.full((1, 52, 3), 2.0)
    monkeypatch.setattr(
        diagnostics,
        "build_constraints",
        lambda *args, **kwargs: (motion_mask, observed_motion),
    )
    monkeypatch.setattr(
        diagnostics,
        "global_end_effector_targets",
        lambda *args, **kwargs: torch.tensor([[1.0, 2.0, 3.0]]),
    )
    native_constraint = object()
    monkeypatch.setattr(
        diagnostics,
        "_native_hand_constraint",
        lambda *args, **kwargs: (native_constraint, "native test constraint"),
    )
    generator = SimpleNamespace(
        motion_history=None,
        device=torch.device("cpu"),
        model=model,
        root_history=torch.tensor([[1.0, 0.8, 3.0], [4.0, 0.8, 6.0]]),
        root_heading=torch.tensor(0.0),
        text_cfg_weight=1.5,
        constraint_cfg_weight=2.5,
        seed=123,
    )

    result = diagnostics.compare_first_generation_ee_conditioning(
        generator,
        torch.arange(4096, dtype=torch.float32),
        (EndEffectorTarget("right_hand", (0.4, 0.2, 0.3)),),
    )

    first_call, second_call, native_call = model.autoregressive_step.call_args_list
    assert first_call.kwargs["motion_mask"] is None
    assert first_call.kwargs["observed_motion"] is None
    assert second_call.kwargs["motion_mask"] is motion_mask
    assert second_call.kwargs["observed_motion"] is observed_motion
    assert first_call.kwargs["cfg_weight"] == second_call.kwargs["cfg_weight"] == (
        1.5,
        2.5,
    )
    torch.testing.assert_close(
        first_call.kwargs["init_global_translation"], torch.tensor([[4.0, 0.0, 6.0]])
    )
    torch.testing.assert_close(
        first_call.kwargs["init_global_translation"],
        second_call.kwargs["init_global_translation"],
    )
    torch.testing.assert_close(
        native_call.kwargs["motion_mask"], torch.ones((1, 52, 3))
    )
    torch.testing.assert_close(
        native_call.kwargs["observed_motion"], torch.full((1, 52, 3), 4.0)
    )
    assert result.unconstrained_motion is first_motion
    assert result.conditioned_motion is second_motion
    assert result.native_motion is native_motion
    assert result.native_constraint_source == "native test constraint"
    torch.testing.assert_close(result.target_positions, torch.tensor([[1.0, 2.0, 3.0]]))


def test_prepare_conditioning_accepts_per_request_embedding() -> None:
    embedding = torch.arange(4096, dtype=torch.float64)
    text_feat, text_pad_mask = prepare_conditioning(
        embedding,
        device=torch.device("cpu"),
    )

    assert text_feat.shape == (1, 1, 4096)
    assert text_feat.dtype is torch.float32
    torch.testing.assert_close(text_feat[0, 0], embedding.float())
    assert text_pad_mask.tolist() == [[True]]
