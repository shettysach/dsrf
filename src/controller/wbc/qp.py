"""Equality-constrained inverse-dynamics solve used by WBC v1."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from controller import DynamicsSnapshot


@dataclass(frozen=True)
class KktSolution:
    qacc: torch.Tensor
    forces: torch.Tensor
    torque: torch.Tensor


def solve_inverse_dynamics(
    dynamics: DynamicsSnapshot,
    qacc_des: torch.Tensor,
    torque_pd: torch.Tensor,
    *,
    acceleration_weight: float,
    pd_weight: float,
    force_weight: float,
) -> KktSolution:
    """Find the closest acceleration/torque satisfying EOM and stance.

    v1 intentionally has equality constraints only.  Friction cones and effort
    bounds require an inequality QP and are a later extension, rather than an
    implicit, unsafe clamp here.
    """
    mass = dynamics.mass_matrix
    nv = mass.shape[0]
    nu = len(dynamics.actuated_dof_indices)
    contacts = dynamics.contacts
    nf = 3 * len(contacts)
    dtype, device = mass.dtype, mass.device
    if qacc_des.shape != (nv,) or torque_pd.shape != (nu,):
        raise ValueError("WBC reference dimensions do not match dynamics snapshot")
    qacc_des = qacc_des.to(dtype=mass.dtype, device=mass.device)
    torque_pd = torque_pd.to(dtype=mass.dtype, device=mass.device)

    jacobian = (
        torch.cat([contact.jacobian for contact in contacts], dim=0)
        if contacts
        else torch.empty((0, nv), dtype=dtype, device=device)
    )
    jdot_qdot = (
        torch.cat([contact.jacobian_dot_velocity for contact in contacts])
        if contacts
        else torch.empty(0, dtype=dtype, device=device)
    )
    selection = torch.zeros((nu, nv), dtype=dtype, device=device)
    selection[
        torch.arange(nu, device=device), dynamics.actuated_dof_indices.long()
    ] = 1.0

    # M qdd + h = J' f + S' tau
    eom = torch.cat((mass, -jacobian.T, -selection.T), dim=1)
    stance = torch.cat(
        (
            jacobian,
            torch.zeros((nf, nf + nu), dtype=dtype, device=device),
        ),
        dim=1,
    )
    equality = torch.cat((eom, stance), dim=0)
    rhs = torch.cat((-dynamics.bias_force, -jdot_qdot))

    diagonal = torch.cat(
        (
            torch.full((nv,), acceleration_weight, dtype=dtype, device=device),
            torch.full((nf,), force_weight, dtype=dtype, device=device),
            torch.full((nu,), pd_weight, dtype=dtype, device=device),
        )
    )
    hessian = torch.diag(diagonal)
    linear = torch.cat(
        (
            -acceleration_weight * qacc_des,
            torch.zeros(nf, dtype=dtype, device=device),
            -pd_weight * torque_pd,
        )
    )
    zeros = torch.zeros(
        (equality.shape[0], equality.shape[0]), dtype=dtype, device=device
    )
    kkt = torch.cat(
        (
            torch.cat((hessian, equality.T), dim=1),
            torch.cat((equality, zeros), dim=1),
        ),
        dim=0,
    )
    solution = torch.linalg.solve(
        kkt, torch.cat((-linear, rhs)),
    )[: nv + nf + nu]
    return KktSolution(
        qacc=solution[:nv],
        forces=solution[nv : nv + nf],
        torque=solution[nv + nf :],
    )
