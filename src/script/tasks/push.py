from __future__ import annotations

from dataclasses import dataclass

from shared.messages import AgentCommand, EndEffectorTarget


@dataclass(frozen=True)
class PushScript:
    """Reach the box, then walk it to the goal with two palm contacts."""

    prompt: str
    box_position: tuple[float, float, float] = (1.65, 0.0, 0.55)
    box_size: tuple[float, float, float] = (0.7, 0.7, 1.1)
    goal_position: tuple[float, float] = (5.0, 0.0)
    robot_position: tuple[float, float, float] = (0.75, 0.0, 0.76)
    hand_spacing: float = 0.32
    # Upper part of the 1.10 m-tall box face. This remains on the physical
    # face (top is z=1.10) while matching ARDY's natural arms-forward height
    # much better than the old center-height target (z=0.55).
    palm_contact_height: float = 1.05
    # Bias the positional constraint just through the face.  ARDY can otherwise
    # stop its hands slightly short of a surface target; collision then resolves
    # this to palm contact without requesting a continuing push.
    contact_depth: float = 0.05

    def next_command(self, observation_id: int) -> AgentCommand | None:
        box_x, _, _ = self.box_position
        robot_x, robot_y, robot_z = self.robot_position
        goal_x, goal_y = self.goal_position
        box_depth, _, _ = self.box_size
        half_spacing = self.hand_spacing / 2.0
        initial_near_face_x = box_x - box_depth / 2.0 + self.contact_depth
        hand_reach_x = initial_near_face_x - robot_x
        final_near_face_x = goal_x - box_depth / 2.0 + self.contact_depth
        final_robot_x = final_near_face_x - hand_reach_x
        match observation_id:
            case 0:
                return AgentCommand(
                    observation_id=observation_id,
                    text=self.prompt,
                    motion="extend both arms straight forward and hold them there",
                    target_xys=(),
                    end_effectors=self._palm_targets(
                        initial_near_face_x,
                        0.0,
                        self.palm_contact_height,
                        robot_x,
                        robot_y,
                        robot_z,
                        half_spacing,
                    ),
                )
            case 1:
                return AgentCommand(
                    observation_id=observation_id,
                    text="Walk forward and push the box onto the green goal.",
                    motion="walk forward while pushing the box with both palms",
                    target_xys=((final_robot_x - robot_x, goal_y - robot_y),),
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
            case _:
                return None

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
