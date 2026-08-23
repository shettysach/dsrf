"""Conversion between SONIC/MJLab actions and physical G1 joint targets."""

from __future__ import annotations

import torch

from shared.g1 import G1_JOINT_COUNT


class G1CommandTransform:
    """An immutable affine transform for the G1's canonical joint order."""

    def __init__(self, default_position: torch.Tensor, scale: torch.Tensor) -> None:
        default_position = default_position.detach().clone().reshape(-1)
        scale = scale.detach().clone().reshape(-1)
        expected = (G1_JOINT_COUNT,)
        if default_position.shape != expected or scale.shape != expected:
            raise ValueError(
                "G1 command transform tensors must both have shape "
                f"{expected}, got {default_position.shape} and {scale.shape}"
            )
        if default_position.device != scale.device:
            raise ValueError(
                "G1 command transform tensors must be on the same device, got "
                f"{default_position.device} and {scale.device}"
            )
        self._default_position = default_position
        self._scale = scale

    @property
    def default_position(self) -> torch.Tensor:
        return self._default_position.clone()

    @property
    def scale(self) -> torch.Tensor:
        return self._scale.clone()

    def decode(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Convert an unbatched residual action into radians."""
        self._validate(raw_action, "raw_action")
        return self._default_position + raw_action * self._scale

    def encode(self, joint_target: torch.Tensor) -> torch.Tensor:
        """Convert an unbatched physical target into a residual action."""
        self._validate(joint_target, "joint_target")
        return (joint_target - self._default_position) / self._scale

    @staticmethod
    def _validate(value: torch.Tensor, name: str) -> None:
        if value.shape != (G1_JOINT_COUNT,):
            raise ValueError(
                f"{name} must have shape ({G1_JOINT_COUNT},), got {value.shape}"
            )
