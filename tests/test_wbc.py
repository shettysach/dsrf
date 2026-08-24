import torch

from controller import ContactDynamics, DynamicsSnapshot
from controller.wbc.qp import solve_inverse_dynamics


def _snapshot() -> DynamicsSnapshot:
    # Two floating-base DoFs, one actuator, and one fixed support direction.
    return DynamicsSnapshot(
        qpos=torch.zeros(3),
        qvel=torch.tensor([0.0, 0.0, 0.2]),
        mass_matrix=torch.eye(3, dtype=torch.float64),
        bias_force=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        contacts=(
            ContactDynamics(
                body="left_ankle_roll_link",
                position_w=torch.zeros(3),
                frame_w=torch.eye(3),
                jacobian=torch.eye(3, dtype=torch.float64),
                jacobian_dot_velocity=torch.zeros(3, dtype=torch.float64),
            ),
        ),
        actuated_dof_indices=torch.tensor([2]),
        joint_stiffness=torch.tensor([100.0], dtype=torch.float64),
        joint_damping=torch.tensor([10.0], dtype=torch.float64),
        effort_limits=torch.tensor([50.0], dtype=torch.float64),
    )


def test_kkt_solution_satisfies_dynamics_and_stance() -> None:
    dynamics = _snapshot()
    solution = solve_inverse_dynamics(
        dynamics,
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0]),
        acceleration_weight=1.0,
        pd_weight=1.0,
        force_weight=1e-4,
    )
    contact = dynamics.contacts[0]
    selection = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    eom = dynamics.mass_matrix @ solution.qacc + dynamics.bias_force
    eom -= contact.jacobian.T @ solution.forces
    eom -= selection.T @ solution.torque
    torch.testing.assert_close(eom, torch.zeros(3, dtype=torch.float64), atol=1e-10, rtol=0)
    torch.testing.assert_close(
        contact.jacobian @ solution.qacc + contact.jacobian_dot_velocity,
        torch.zeros(3, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


def test_pd_target_reconstructs_requested_torque() -> None:
    q, qdot = torch.tensor([0.2]), torch.tensor([0.3])
    qdot_ref, torque, kp, kd = torch.tensor([0.7]), torch.tensor([8.0]), torch.tensor([100.0]), torch.tensor([10.0])
    command = q + (torque - kd * (qdot_ref - qdot)) / kp
    reconstructed = kp * (command - q) + kd * (qdot_ref - qdot)
    torch.testing.assert_close(reconstructed, torque)
