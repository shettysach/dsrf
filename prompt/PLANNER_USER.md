Choose one motion and direction for the robot's next action.

The motion value must be stand or walk.
The direction value must be forward, backward, left, or right.
With stand, direction is the desired robot-facing direction; use it to rotate in place.
With walk, left and right move laterally without rotating the robot.
Call robot_action with the motion and direction; do not return a text command.
Invoke robot_action exactly once for each observation. Never emit multiple
robot_action calls in one response; wait for the next observation before
choosing another motion.
