# Task: Sokoban

## Objective

Push both yellow boxes onto the two green goal regions.

A box is complete as soon as any part of it touches green. It does not need to be
centered or fully inside the goal. Once a box touches green, stop pushing it and
never touch it again.

When both boxes touch separate green goals, issue **stand**.

## Scene and controls

- Yellow cubes are movable boxes.
- Green floor regions are goals.
- Walking into a box pushes it in the direction of travel.
- Boxes are light and easy to push.
- Boxes can be pushed but cannot be pulled.
- Directions are relative to the robot's current facing direction.
- Turning rotates both the robot and the camera.
- A 2D waypoint makes the robot walk toward a visible point on the floor.

## Main strategy

Focus on only one unfinished box at a time.

1. Look for a visible yellow box and a reachable green goal.
2. Choose one box to complete first. Ignore the other box while working on it.
3. Determine which direction the chosen box must move to reach green.
4. Approach from the side opposite the goal. Exact centering is unnecessary; only
   avoid pushing the box in a clearly wrong direction.
5. Push toward the goal with one short direction command at a time.
6. Inspect the next image after every push near green.
7. As soon as any part of the box touches green, that box is complete. Do not push
   it farther or try to center it.
8. Move away from the completed box and focus only on the remaining box.

There is no fixed box order, box-goal assignment, or required push direction. Use
the current image to choose the simplest safe solution.

## Finding the remaining box

After completing the first box:

1. Never touch the completed box again.
2. If the unfinished box is not visible, turn left or right once.
3. Inspect the new image after every turn. Continue exploring one turn at a time
   until the unfinished box or an open route toward it is visible.
4. If it is nearby, approach with short direction commands.
5. If it is far away or would require many small steps, use a 2D waypoint on
   visible open floor near the useful side of the box.
6. Never place a waypoint on a box, wall, robot, or other obstacle. Never choose a
   waypoint whose route passes through a box.
7. Leave some space before the box. After the waypoint, inspect again and use
   direction commands for the final approach and all pushing.

## Pushing rules

- Push only the currently selected unfinished box.
- Push toward a visible green goal, not toward an ordinary wall or corner.
- Use direction commands, not 2D waypoints, to contact and push a box.
- Do not continue a push after the box first touches green.
- Do not try to improve, center, or reposition a completed box.
- Keep an unfinished box away from the wrong wall because it cannot be pulled back.

## Recovery

If an unfinished box moves in the wrong direction:

1. Stop pushing.
2. Move away from the box to create space.
3. Walk around it through visible open floor.
4. Approach from a side that will push it toward green.
5. Resume with one short push and inspect again.

Never use recovery on a box that already touches green.

## Finish

Issue **stand** when both yellow boxes visibly touch separate green regions. Edge
contact is sufficient. If only one box touches green, ignore it and continue with
the unfinished box.
