import pytest
import torch

from sim.virtual_force import VirtualForce, _derive_hand_motion, _HandMotion


def _virtual_force(*, magnitude: float = 40.0, maximum: float = 50.0) -> VirtualForce:
    virtual_force = VirtualForce(
        ("box",),
        dt=0.02,
        device="cpu",
        magnitude=magnitude,
        maximum=maximum,
        min_hand_speed=0.05,
        lookahead_frames=1,
        ramp_frames=4,
    )
    direction = torch.tensor((1.0, 0.0, 0.0))
    moving = _HandMotion(
        directions=direction.repeat(8, 1),
        speeds=torch.ones(8),
    )
    virtual_force._hand_motion = {"left_hand": moving, "right_hand": moving}
    return virtual_force


def test_virtual_force_requires_contact_and_ramps_from_zero() -> None:
    virtual_force = _virtual_force()

    no_contact = virtual_force.compute(0, set())
    first_contact = virtual_force.compute(0, {("right_hand", "box")})
    second_contact = virtual_force.compute(1, {("right_hand", "box")})

    torch.testing.assert_close(no_contact.forces["box"], torch.zeros(3))
    torch.testing.assert_close(first_contact.forces["box"], torch.zeros(3))
    torch.testing.assert_close(
        second_contact.forces["box"], torch.tensor((10.0, 0.0, 0.0))
    )
    assert first_contact.started_contacts == (("right_hand", "box"),)


def test_virtual_force_sums_hands_clamps_per_object_and_resets_on_separation() -> None:
    virtual_force = _virtual_force()
    contacts = {("left_hand", "box"), ("right_hand", "box")}

    virtual_force.compute(0, contacts)
    virtual_force.compute(1, contacts)
    virtual_force.compute(2, contacts)
    clamped = virtual_force.compute(3, contacts)
    separated = virtual_force.compute(4, set())

    assert torch.linalg.vector_norm(clamped.forces["box"]).item() == pytest.approx(
        50.0
    )
    torch.testing.assert_close(separated.forces["box"], torch.zeros(3))
    assert separated.ended_contacts == (
        ("left_hand", "box"),
        ("right_hand", "box"),
    )


def test_hand_motion_uses_actual_tail_lookahead_and_zeroes_final_direction() -> None:
    positions = torch.tensor(((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0)))

    motion = _derive_hand_motion(positions, dt=0.1, lookahead_frames=2)

    torch.testing.assert_close(motion.speeds, torch.tensor((1.5, 1.0, 0.0)))
    torch.testing.assert_close(motion.directions[-1], torch.zeros(3))
