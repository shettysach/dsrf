from __future__ import annotations

from dataclasses import dataclass

from shared.messages import AgentCommand, EndEffectorTarget


@dataclass(frozen=True)
class PushScript:
    """One deterministic two-palm push command for the aligned box task."""

    prompt: str
    box_position: tuple[float, float, float] = (1.55, 0.0, 0.55)
    box_size: tuple[float, float, float] = (0.5, 0.7, 1.1)
    robot_position: tuple[float, float, float] = (0.95, 0.0, 0.76)
    hand_spacing: float = 0.32
    # Bias the positional constraint just through the face.  ARDY can otherwise
    # stop its hands slightly short of a surface target; collision then resolves
    # this to palm contact without requesting a continuing push.
    contact_depth: float = 0.05

    def next_command(self, observation_id: int) -> AgentCommand | None:
        # This initial experiment has exactly one phase.  Future phases belong
        # here, rather than in the generic TaskScript interface.
        if observation_id != 0:
            return None

        box_x, box_y, box_z = self.box_position
        robot_x, robot_y, robot_z = self.robot_position
        box_depth, _, _ = self.box_size
        half_spacing = self.hand_spacing / 2.0
        near_face_x = box_x - box_depth / 2.0 + self.contact_depth
        return AgentCommand(
            observation_id=observation_id,
            text=self.prompt,
            motion="face both palms outward, then reach forward to touch",
            target_xys=(),
            end_effectors=(
                EndEffectorTarget(
                    "left_hand",
                    (
                        near_face_x - robot_x,
                        box_y + half_spacing - robot_y,
                        box_z - robot_z,
                    ),
                ),
                EndEffectorTarget(
                    "right_hand",
                    (
                        near_face_x - robot_x,
                        box_y - half_spacing - robot_y,
                        box_z - robot_z,
                    ),
                ),
            ),
        )
