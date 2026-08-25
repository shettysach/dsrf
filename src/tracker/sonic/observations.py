from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

type G1ReferenceStep = Literal[1, 5]

OBSERVATION_DIMS = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
    "motion_joint_positions_10frame_step1": 290,
    "motion_joint_velocities_10frame_step1": 290,
    "motion_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_lowerbody_10frame_step1": 120,
    "motion_joint_velocities_lowerbody_10frame_step1": 120,
    "smpl_joints_4frame_step1": 288,
    "smpl_anchor_orientation_4frame_step1": 24,
    "motion_joint_positions_wrists_4frame_step1": 24,
}


def _g1_names(step: G1ReferenceStep) -> frozenset[str]:
    suffix = f"10frame_step{step}"
    return frozenset(
        {
            "encoder_mode_4",
            f"motion_joint_positions_{suffix}",
            f"motion_joint_velocities_{suffix}",
            f"motion_anchor_orientation_{suffix}",
        }
    )


G1_NAMES: dict[G1ReferenceStep, frozenset[str]] = {
    1: _g1_names(1),
    5: _g1_names(5),
}


@dataclass(frozen=True)
class ObservationLayout:
    policy_slices: dict[str, slice]
    encoder_slices: dict[str, slice]
    policy_input_dimension: int
    encoder_input_dimension: int
    encoder_dimension: int
    g1_step: G1ReferenceStep

    @classmethod
    def load(cls, path: Path) -> "ObservationLayout":
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        encoder = document["encoder"]
        policy_names = _enabled_names(document["observations"])
        encoder_names = _enabled_names(encoder["encoder_observations"])
        g1_mode = next(
            mode for mode in encoder["encoder_modes"] if mode["name"] == "g1"
        )
        required = frozenset(g1_mode["required_observations"])
        g1_step = next(step for step, names in G1_NAMES.items() if required == names)

        policy_dimension = sum(OBSERVATION_DIMS[name] for name in policy_names)
        encoder_dimension = int(encoder["dimension"])
        return cls(
            policy_slices=_observation_slices(policy_names),
            encoder_slices=_observation_slices(encoder_names),
            policy_input_dimension=policy_dimension,
            encoder_input_dimension=sum(OBSERVATION_DIMS[n] for n in encoder_names),
            encoder_dimension=encoder_dimension,
            g1_step=g1_step,
        )


def _enabled_names(value: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(str(item["name"]) for item in value if item.get("enabled", False))


def _observation_slices(names: tuple[str, ...]) -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for name in names:
        dimension = OBSERVATION_DIMS[name]
        result[name] = slice(offset, offset + dimension)
        offset += dimension
    return result
