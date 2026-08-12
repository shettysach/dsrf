from __future__ import annotations

from tasks.spec import SceneSpecFn, TaskSpec, ViewerSpec


def _make_scene(*, goal_index: int | None = None) -> SceneSpecFn:
    del goal_index  # This task has one fixed layout.
    from tasks.see_saw.scene import make_see_saw_spec_fn

    return make_see_saw_spec_fn()


TASK = TaskSpec(
    name="see-saw",
    objective="Walk onto the counterweighted see-saw and stand where it balances.",
    make_scene=_make_scene,
    viewer=ViewerSpec(distance=4.5, elevation=-24.0),
)
