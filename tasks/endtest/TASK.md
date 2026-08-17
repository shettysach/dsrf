# Task: endtest

## Objective

Place the left foot on the green square ahead-left of the robot, then place the
right foot on the green square ahead-right. The two small green squares are
visual foot-placement targets on otherwise empty floor.

## Procedure

1. Keep the robot in place. Do not use `waypoints_2d` or a walking command.
2. Use only a `left_foot` end-effector target, placed on the center of the
   visible left green square. Use a concise stepping motion such as "step the
   left foot onto the left green square".
3. Reassess the image. Once the left foot is inside its green square, use only
   a `right_foot` end-effector target on the center of the right green square.
   Use a concise stepping motion such as "step the right foot onto the right
   green square".

## Safety

- Test one foot per command; never constrain both feet in the same command.
- Target the middle of each green square, not its edge or nearby floor.
- Do not use hand targets in this task.

## Finish

Stop only when both feet are visibly placed on their corresponding green
squares.
