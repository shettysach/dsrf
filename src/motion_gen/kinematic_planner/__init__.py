# https://nvlabs.github.io/GR00T-WholeBodyControl/references/planner_onnx.html

from motion_gen.kinematic_planner.parser import (
    KinematicPlannerCommand,
    PlannerMode,
    parse_kinematic_planner_command,
    planner_direction,
    planner_mode,
)

__all__ = [
    "PlannerMode",
    "KinematicPlanner",
    "KinematicPlannerCommand",
    "parse_kinematic_planner_command",
    "planner_direction",
    "planner_mode",
]


def __getattr__(name: str):
    if name == "KinematicPlanner":
        from motion_gen.kinematic_planner.generator import KinematicPlanner

        return KinematicPlanner
    raise AttributeError(name)
