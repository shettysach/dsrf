"""Explicit source-frame constraints for a single native generation window."""

from dataclasses import dataclass

from shared.messages import EndEffectorTarget


@dataclass(frozen=True)
class TimedTargets:
    # Zero-based NEW frame index. Positions are relative to the observed root/yaw.
    frame: int
    root_xy: tuple[float, float]
    end_effectors: tuple[EndEffectorTarget, ...] = ()
    root_upright: bool = False
