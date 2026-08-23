from pathlib import Path

import yaml

from controller.sonic.observations import ObservationLayout


def _write_layout(path: Path, *, step: int) -> Path:
    suffix = f"10frame_step{step}"
    required = [
        "encoder_mode_4",
        f"motion_joint_positions_{suffix}",
        f"motion_joint_velocities_{suffix}",
        f"motion_anchor_orientation_{suffix}",
    ]
    document = {
        "observations": [
            {"name": "token_state", "enabled": True},
            {
                "name": "his_base_angular_velocity_10frame_step1",
                "enabled": True,
            },
        ],
        "encoder": {
            "dimension": 64,
            "encoder_observations": [
                {"name": name, "enabled": True} for name in required
            ],
            "encoder_modes": [
                {
                    "name": "g1",
                    "required_observations": required,
                }
            ],
        },
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_default_observation_layout(tmp_path: Path) -> None:
    layout = ObservationLayout.load(_write_layout(tmp_path / "default.yml", step=5))
    assert layout.encoder_input_dimension == 644
    assert layout.policy_input_dimension == 94
    assert layout.g1_step == 5


def test_low_latency_observation_layout(tmp_path: Path) -> None:
    layout = ObservationLayout.load(_write_layout(tmp_path / "low-latency.yml", step=1))
    assert layout.encoder_input_dimension == 644
    assert layout.policy_input_dimension == 94
    assert layout.g1_step == 1
