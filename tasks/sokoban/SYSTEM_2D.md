# Task: Sokoban

## Objective

Push both yellow boxes completely onto their green goal regions.

The arena is asymmetric. Remember this exact box-goal map:

- **Lower-left box → far-wall green goal → push forward.**
- **Upper-right box → right-wall green goal → push right.**

Complete the far-wall box first. When both boxes are fully on separate green goals,
issue **stand**.

## Non-negotiable rules

1. Align before touching a box. The robot center, box center, and goal center must
   form one straight line in the intended push direction.
2. Push through the center, never through a corner or diagonally.
3. Use 2D waypoints only for traveling across visible open floor. Never use a
   waypoint to contact or push a box.
4. Inspect the new image after every action. Do not blindly repeat commands near a
   box or goal.
5. A box fully inside its green goal is **locked and finished**. Never touch, push,
   align against, or correct it again.
6. Boxes cannot be pulled. Keep each unfinished box away from every wall except its
   own wall-touching goal.

## Stage 1: far-wall box

Work only on the lower-left box.

1. Stay behind the boxes and move left through open floor into the lower-left
   box's lane.
2. Stop with a clear gap before the box. Do not touch it while moving into position.
3. Face the far wall. Stand directly behind the box so the robot, box, and far-wall
   goal are centered on one forward line.
4. If alignment is wrong, back away and correct it before contact.
5. Once aligned, push only **forward**.
6. Near the green goal, check the box before every additional forward command.
7. The instant the entire box is inside the green region, stop pushing. Do not add a
   centering push and do not push it toward the wall.
8. Walk **backward** until clearly separated from the completed box.

## Stage 2: find and approach the right-wall box

Now ignore the completed far-wall box permanently. Work only on the upper-right
box.

1. Return through the open center without touching the completed box.
2. If the unfinished box is not visible, issue one turn left or turn right, then
   inspect the next image. Turn one step at a time until the unfinished box is in
   view.
3. After finding it, restore the useful arena orientation: face the far wall so the
   right wall and the box's green goal are to the robot's right. Directions are
   relative to the robot, so **right** is correct only in this orientation.
4. The required pushing position is directly to the **left of the box at the same
   depth**. The robot, box, and right-wall goal must form one horizontal line.
5. If that position is nearby, approach it with short direction commands.
6. If it is far away or would take many direction commands, choose a 2D waypoint on
   visible open floor near the box's left side. The route to it must be clear.
7. Never choose the box, a wall, an obstacle, or floor beyond the box as a waypoint.
   Leave a visible gap from the box. After reaching the waypoint, inspect again.
8. Stop using waypoints before contact. Use short direction commands for final
   positioning and alignment.

## Stage 3: right-wall box

1. Face the far wall and stand directly left of the unfinished box at the same
   depth.
2. Verify that the robot center, box center, and right-wall goal center form one
   straight line to the right.
3. Once aligned, push only **right**. Do not turn and do not push forward.
4. Near the green goal, check the box before every additional right command.
5. The instant the entire box is inside the green region, stop pushing. Do not add a
   centering push and do not push it into the wall.
6. Move **left** to separate from the completed box.

## Recovery

If an unfinished box moves off its box-goal line or is contacted off-center:

1. Stop pushing immediately.
2. Separate from it: move backward after a forward push, or left after a right push.
3. Return to the correct side: behind the far-wall box, or left of the right-wall
   box.
4. Rebuild the straight robot-box-goal centerline before pushing again.

Never apply recovery to a completed box. A completed box remains locked.

## Finish

Issue **stand** only after visually verifying all three facts:

- Both yellow boxes are fully inside green regions.
- The two boxes occupy different goals.
- Neither green goal is empty or merely touched at its edge.

If only one box is complete, continue working exclusively on the unfinished box.
