"""Run a controlled first-window ARDY EE-conditioning A/B comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from motion_gen.ardy.diagnostics import compare_first_generation_ee_conditioning
from motion_gen.ardy.generator import Ardy
from motion_gen.ardy.text_encoder import TextEncoder
from shared.messages import EndEffectorTarget

_JOINT_NAMES = {
    "left_hand": "left_hand_roll_skel",
    "right_hand": "right_hand_roll_skel",
}
_DEFAULT_CHECKPOINTS_DIR = Path("/tmp/checkpoints")
_DEFAULT_TEXT_ENCODER_MODEL = Path("/home/sach/Desktop/encoding/model")
_DEFAULT_PROMPT = "extend both arms straight forward and hold them there"
_DEFAULT_RIGHT_HAND_TARGET = (0.58, -0.16, 0.40)


def main() -> None:
    args = _arguments()
    generator = Ardy(
        args.checkpoints_dir,
        device=args.device,
        text_cfg_weight=args.text_cfg_weight,
        constraint_cfg_weight=args.constraint_cfg_weight,
        seed=args.seed,
    )
    embedding = TextEncoder(args.text_encoder_model, device=args.device).encode(args.text)
    target = EndEffectorTarget(args.end_effector, tuple(args.target_xyz))
    result = compare_first_generation_ee_conditioning(generator, embedding, (target,))
    _print_result(generator, target, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text": args.text,
            "end_effector": args.end_effector,
            "target_xyz": args.target_xyz,
            "unconstrained_motion": result.unconstrained_motion.cpu(),
            "conditioned_motion": result.conditioned_motion.cpu(),
            "native_motion": result.native_motion.cpu(),
            "motion_mask": result.motion_mask.cpu(),
            "observed_motion": result.observed_motion.cpu(),
            "native_motion_mask": result.native_motion_mask.cpu(),
            "native_observed_motion": result.native_observed_motion.cpu(),
            "unconstrained_decoded": _cpu_tensors(result.unconstrained_decoded),
            "conditioned_decoded": _cpu_tensors(result.conditioned_decoded),
            "native_decoded": _cpu_tensors(result.native_decoded),
            "native_constraint_source": result.native_constraint_source,
            "target_positions": result.target_positions.cpu(),
        },
        args.output,
    )
    print(f"Saved raw comparison tensors to {args.output}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults intentionally match arms_forward_constrained.yml.
    parser.add_argument("--checkpoints-dir", type=Path, default=_DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument(
        "--text-encoder-model", type=Path, default=_DEFAULT_TEXT_ENCODER_MODEL
    )
    parser.add_argument("--text", default=_DEFAULT_PROMPT)
    parser.add_argument("--end-effector", choices=tuple(_JOINT_NAMES), default="right_hand")
    parser.add_argument("--target-xyz", type=float, nargs=3, default=_DEFAULT_RIGHT_HAND_TARGET)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-cfg-weight", type=float, default=2.0)
    parser.add_argument("--constraint-cfg-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("out/ardy-ee-ab.pt"))
    return parser.parse_args()


def _print_result(generator: Ardy, target: EndEffectorTarget, result) -> None:
    joint = generator.model.skeleton.bone_order_names.index(_JOINT_NAMES[target.name])
    requested = result.target_positions[0]
    for label, decoded in (
        ("unconstrained", result.unconstrained_decoded),
        ("legacy_sparse", result.conditioned_decoded),
        ("native_requested_xyz", result.native_decoded),
    ):
        endpoint = decoded["posed_joints"][0, -1, joint]
        error = torch.linalg.vector_norm(endpoint - requested)
        print(f"{label}_endpoint={endpoint.tolist()} error_m={float(error):.4f}")
    feature_delta = torch.linalg.vector_norm(
        result.conditioned_motion - result.unconstrained_motion, dim=-1
    )
    print(
        f"mask_features={int(result.motion_mask.sum())} "
        f"native_mask_features={int(result.native_motion_mask.sum())} "
        f"native_constraint_source={result.native_constraint_source} "
        f"feature_delta_final={float(feature_delta[0, -1]):.6f}"
    )


def _cpu_tensors(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.cpu() for name, value in values.items()}


if __name__ == "__main__":
    main()
