from __future__ import annotations

from tasks.spec import SceneSpecFn, TaskSpec, ViewerSpec


def _make_scene(*, goal_index: int | None = None) -> SceneSpecFn:
    del goal_index  # This task has one fixed layout.
    from tasks.stack_steps.scene import make_stack_steps_spec_fn

    return make_stack_steps_spec_fn()


TASK = TaskSpec(
    name="stack-steps",
    objective="Stack the two blocks into steps and reach the green platform.",
    make_scene=_make_scene,
    viewer=ViewerSpec(distance=4.0, elevation=-28.0),
)
