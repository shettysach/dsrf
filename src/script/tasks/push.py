from __future__ import annotations

from dataclasses import dataclass

from shared.messages import AgentCommand, EndEffectorTarget


@dataclass(frozen=True)
class PushScript:
    """Push the nearby aligned box in one continuous two-palm motion."""

    prompt: str
    box_position: tuple[float, float, float] = (1.60, 0.0, 0.55)
    box_size: tuple[float, float, float] = (0.9, 0.9, 1.1)
    goal_position: tuple[float, float] = (1.90, 0.0)
    robot_position: tuple[float, float, float] = (0.75, 0.0, 0.76)
    hand_spacing: float = 0.36
    palm_contact_height: float = 1.0
    # Bias the positional constraint just through the face.  ARDY can otherwise
    # stop its hands slightly short of a surface target; collision then resolves
    # this to palm contact without requesting a continuing push.
    contact_depth: float = 0.05

    def next_command(self, observation_id: int) -> AgentCommand | None:
        if observation_id != 0:
            return None

        box_x, _, _ = self.box_position
        robot_x, robot_y, robot_z = self.robot_position
        goal_x, goal_y = self.goal_position
        box_depth, _, _ = self.box_size
        half_spacing = self.hand_spacing / 2.0
        push_distance = goal_x - box_x
        final_near_face_x = goal_x - box_depth / 2.0 + self.contact_depth
        return AgentCommand(
            observation_id=observation_id,
            text=self.prompt,
            motion=self.prompt,
            target_xys=((push_distance, goal_y - robot_y),),
            end_effectors=self._palm_targets(
                final_near_face_x,
                goal_y,
                self.palm_contact_height,
                robot_x,
                robot_y,
                robot_z,
                half_spacing,
            ),
        )

    @staticmethod
    def _palm_targets(
        contact_x: float,
        contact_y: float,
        contact_z: float,
        robot_x: float,
        robot_y: float,
        robot_z: float,
        half_spacing: float,
    ) -> tuple[EndEffectorTarget, EndEffectorTarget]:
        return (
            EndEffectorTarget(
                "left_hand",
                (
                    contact_x - robot_x,
                    contact_y + half_spacing - robot_y,
                    contact_z - robot_z,
                ),
            ),
            EndEffectorTarget(
                "right_hand",
                (
                    contact_x - robot_x,
                    contact_y - half_spacing - robot_y,
                    contact_z - robot_z,
                ),
            ),
        )
