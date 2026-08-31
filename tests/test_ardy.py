from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from ardy.exports.mujoco import MujocoQposConverter
from ardy.skeleton import G1Skeleton34

from motion_gen.ardy.encoder import prepare_conditioning
from motion_gen.ardy.history import qpos_to_ardy_inputs
from shared.g1 import standing_qpos


def test_ardy_model_loader_receives_a_device_string(monkeypatch, tmp_path) -> None:
    import motion_gen.ardy.generator as ardy_generator

    received: dict[str, object] = {}
    model = SimpleNamespace(
        motion_rep=SimpleNamespace(fps=25),
        skeleton=object(),
        num_frames_per_token=8,
    )
    monkeypatch.setattr(
        ardy_generator,
        "load_model",
        lambda *args, **kwargs: received.update(kwargs) or model,
    )
    monkeypatch.setattr(ardy_generator, "MujocoQposConverter", lambda _: object())
    monkeypatch.setattr(
        ardy_generator,
        "build_initial_history",
        lambda *args, **kwargs: torch.zeros((1, 4, 1)),
    )
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
    assert generator.history_frames == 4
    assert generator.initial_history is None


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


def test_ardy_history_conditions_generation_but_is_not_returned() -> None:
    import motion_gen.ardy.generator as ardy_generator

    model = Mock()
    model.gen_horizon_len = 52
    model.diffusion.num_base_steps = 10
    returned_motion = torch.arange(129 * 3, dtype=torch.float32).reshape(1, 129, 3)
    model.return_value = returned_motion
    model.motion_rep.create_conditions.return_value = (
        torch.zeros((129, 3)),
        torch.zeros((129, 3)),
    )
    model.motion_rep.inverse.side_effect = lambda motion, **kwargs: {
        "motion": motion,
        "root_positions": torch.zeros((1, 125, 3)),
        "global_root_heading": torch.tensor([[[1.0, 0.0]] * 124 + [[0.0, 1.0]]]),
    }
    converter = Mock()
    converter.dict_to_qpos.return_value = torch.zeros((1, 125, 36))

    generator = ardy_generator.Ardy.__new__(ardy_generator.Ardy)
    generator.device = torch.device("cpu")
    generator.model = model
    generator.converter = converter
    generator.history_frames = 4
    generator.initial_history = torch.zeros((1, 4, 3))
    generator.root_history = torch.zeros((2, 3))
    generator.root_heading = torch.tensor(0.0)
    initial_history = generator.initial_history

    embedding = torch.arange(4096, dtype=torch.float32)
    qpos = generator.generate(embedding, ((0.0, 0.5),))

    assert qpos.shape == (125, 36)
    assert converter.dict_to_qpos.call_args.kwargs["numpy"] is False
    model.motion_rep.inverse.assert_called_once()
    generated_motion = model.motion_rep.inverse.call_args.args[0]
    assert generated_motion.shape == (1, 125, 3)
    assert model.call_args.kwargs["init_history_sequence"] is initial_history
    assert model.call_args.kwargs["first_heading_angle"] is None
    assert model.call_args.kwargs["motion_mask"] is not None
    assert model.call_args.kwargs["observed_motion"] is not None
    assert model.call_args.kwargs["text_feat"].shape == (1, 1, 4096)
    torch.testing.assert_close(model.call_args.kwargs["text_feat"][0, 0], embedding)
    torch.testing.assert_close(generator.initial_history, returned_motion[:, -4:])
    torch.testing.assert_close(generator.root_history, torch.zeros((2, 3)))
    torch.testing.assert_close(generator.root_heading, torch.tensor(torch.pi / 2.0))


def test_unconstrained_ardy_uses_hf_style_rolling_four_frame_history() -> None:
    import motion_gen.ardy.generator as ardy_generator

    model = Mock()
    model.gen_horizon_len = 52
    model.diffusion.num_base_steps = 10
    model.autoregressive_step.side_effect = lambda **kwargs: torch.zeros(
        (1, kwargs["num_frames"], 3)
    )
    model.motion_rep.inverse.return_value = {
        "root_positions": torch.zeros((1, 125, 3)),
        "global_root_heading": torch.tensor([[[1.0, 0.0]] * 125]),
    }
    converter = Mock()
    converter.dict_to_qpos.return_value = torch.zeros((1, 125, 36))

    generator = ardy_generator.Ardy.__new__(ardy_generator.Ardy)
    generator.device = torch.device("cpu")
    generator.model = model
    generator.converter = converter
    generator.history_frames = 4
    generator.initial_history = torch.zeros((1, 4, 3))
    generator.root_history = torch.zeros((2, 3))
    generator.root_heading = torch.tensor(0.0)

    generator.generate(torch.arange(4096, dtype=torch.float32), ())

    assert model.motion_rep.create_conditions.call_count == 0
    assert model.call_count == 0
    assert model.autoregressive_step.call_count == 3
    calls = model.autoregressive_step.call_args_list
    assert calls[0].kwargs["num_frames"] == 52
    assert calls[0].kwargs["init_history_sequence"] is None
    assert calls[1].kwargs["num_frames"] == 56
    assert calls[1].kwargs["init_history_sequence"].shape == (1, 4, 3)
    assert calls[2].kwargs["init_history_sequence"].shape == (1, 4, 3)
    assert all(call.kwargs["cfg_weight"] == 2.0 for call in calls)


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
