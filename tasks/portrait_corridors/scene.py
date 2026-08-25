from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from mujoco import MjSpec  # ty: ignore[unresolved-import]

MJGEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX  # ty: ignore[unresolved-attribute]
MJGEOM_MESH = mujoco.mjtGeom.mjGEOM_MESH  # ty: ignore[unresolved-attribute]
MJMESH_INERTIA_SHELL = (
    mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL  # ty: ignore[unresolved-attribute]
)
MJTEXTURE_2D = mujoco.mjtTexture.mjTEXTURE_2D  # ty: ignore[unresolved-attribute]

WALL_SOLREF: tuple[float, float] = (0.05, 2.0)
WALL_SOLIMP: tuple[float, float, float, float, float] = (
    0.7,
    0.95,
    0.03,
    0.5,
    2.0,
)

_IMAGES_DIR = Path(__file__).resolve().parent / "images"
_FORWARD_CAMERA_QUAT = (-0.5, -0.5, 0.5, 0.5)


@dataclass(frozen=True)
class _Portrait:
    name: str
    half_size: tuple[float, float] = (0.8, 1.1)


_PORTRAITS = (
    _Portrait("linus"),
    _Portrait("jobs"),
    _Portrait("nolan"),
)


def make_portrait_corridors_spec_fn(
    *,
    seed: int | None = None,
    start_x: float = -2.0,
    corridor_length: float = 8.0,
    corridor_width: float = 2.0,
    divider_start_x: float | None = None,
    wall_height: float = 2.5,
    wall_thickness: float = 0.2,
    wall_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    portrait_wall_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> Callable[["MjSpec"], None]:
    """Create three corridors with a randomly assigned portrait in each one."""

    portraits = list(_PORTRAITS)
    random.Random(seed).shuffle(portraits)

    def add_portrait_corridors(spec: MjSpec) -> None:
        end_x = start_x + corridor_length
        resolved_divider_start_x = (
            start_x + corridor_length * 0.5
            if divider_start_x is None
            else divider_start_x
        )
        half_total_width = corridor_width * 1.5
        half_wall_height = wall_height * 0.5
        half_wall_thickness = wall_thickness * 0.5
        center_x = (start_x + end_x) * 0.5
        wall_z = half_wall_height

        _add_wall(
            spec,
            name="portrait_corridors_end_wall",
            pos=(end_x, 0.0, wall_z),
            size=(half_wall_thickness, half_total_width, half_wall_height),
            rgba=portrait_wall_rgba,
        )
        for side, y in (("north", half_total_width), ("south", -half_total_width)):
            _add_wall(
                spec,
                name=f"portrait_corridors_{side}_wall",
                pos=(center_x, y, wall_z),
                size=(corridor_length * 0.5, half_wall_thickness, half_wall_height),
                rgba=wall_rgba,
            )

        # The dividers begin ahead of the robot, leaving room to choose a lane.
        divider_center_x = (resolved_divider_start_x + end_x) * 0.5
        divider_half_length = (end_x - resolved_divider_start_x) * 0.5
        for index, y in enumerate((-corridor_width * 0.5, corridor_width * 0.5), 1):
            _add_wall(
                spec,
                name=f"portrait_corridors_divider_{index}_wall",
                pos=(divider_center_x, y, wall_z),
                size=(divider_half_length, half_wall_thickness, half_wall_height),
                rgba=wall_rgba,
            )

        camera_x = resolved_divider_start_x - 0.2
        for corridor, y in (
            ("left", corridor_width),
            ("center", 0.0),
            ("right", -corridor_width),
        ):
            camera = spec.worldbody.add_camera(name=f"corridor_{corridor}")
            camera.pos = (camera_x, y, 1.25)
            camera.quat = _FORWARD_CAMERA_QUAT
            camera.fovy = 65.0

        portrait_x = end_x - half_wall_thickness - 0.01
        corridor_positions = (
            (portrait_x, corridor_width, 1.25),
            (portrait_x, 0.0, 1.25),
            (portrait_x, -corridor_width, 1.25),
        )
        for portrait, pos in zip(portraits, corridor_positions, strict=True):
            _add_portrait(
                spec,
                name=portrait.name,
                pos=pos,
                half_size=portrait.half_size,
            )

    return add_portrait_corridors


def _add_wall(
    spec: "MjSpec",
    *,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> None:
    body = spec.worldbody.add_body(name=name)
    body.pos = pos
    body.add_geom(
        name=f"{name}_collision",
        type=MJGEOM_BOX,
        size=size,
        rgba=rgba,
        contype=1,
        conaffinity=1,
        solref=WALL_SOLREF,
        solimp=WALL_SOLIMP,
    )


def _add_portrait(
    spec: "MjSpec",
    *,
    name: str,
    pos: tuple[float, float, float],
    half_size: tuple[float, float] = (0.8, 1.1),
) -> None:
    texture = spec.add_texture(
        name=f"portrait_corridors_{name}_texture",
        type=MJTEXTURE_2D,
        file=str(_IMAGES_DIR / f"{name}.png"),
    )
    material = spec.add_material(name=f"portrait_corridors_{name}_material")
    material.textures[1] = texture.name
    material.texrepeat = (1.0, 1.0)
    material.emission = 0.15

    # A single quad gives the portrait one direct UV map.
    half_width, half_height = half_size
    mesh = spec.add_mesh(
        name=f"portrait_corridors_{name}_mesh",
        inertia=MJMESH_INERTIA_SHELL,
    )
    mesh.uservert = [
        0.0,
        -half_width,
        -half_height,
        0.0,
        half_width,
        -half_height,
        0.0,
        half_width,
        half_height,
        0.0,
        -half_width,
        half_height,
    ]
    mesh.userface = [0, 2, 1, 0, 3, 2]
    mesh.usertexcoord = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    mesh.userfacetexcoord = [0, 2, 1, 0, 3, 2]

    body = spec.worldbody.add_body(name=f"portrait_corridors_{name}_portrait")
    body.pos = pos
    body.add_geom(
        name=f"portrait_corridors_{name}_portrait_visual",
        type=MJGEOM_MESH,
        meshname=mesh.name,
        material=material.name,
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
