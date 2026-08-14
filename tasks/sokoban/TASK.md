# Task: Sokoban

## Objective

Push both yellow boxes completely onto their matching green floor goals, one box
at a time. When both boxes are fully centered on separate goals, issue **stand**.

This arena has two box-goal lanes:

- The **lower-left box** is already aligned with the green goal against the far
  wall. It must move only **forward**.
- The **upper-right box** is already aligned with the green goal against the
  right wall. It must move only **right**.

The boxes do not need sideways correction. Never push either box out of its
existing lane.

## Controls and scene

- Yellow cubes are pushable boxes; green floor regions are goals.
- Walking into a box pushes it. Boxes can be pushed but cannot be pulled.
- **forward** and **backward** move along the robot's facing direction.
- **left** and **right** move laterally without rotating the robot or camera.
- Dark walls bound the arena. The open central floor is the safe route between
  the two box lanes.

Use open floor for repositioning. Never place a movement target on a box, goal,
wall, or other obstacle, and do not route through either box.

## Required push alignment

Before touching a box, establish this exact straight-line arrangement:

**ROBOT → BOX → GOAL**

Stand on the side of the box opposite its goal. The robot's center, the box's
center, and the goal's center must lie on the same line. Look through the center
of the box toward the center of the goal.

Being near a box is not enough. If the box is not directly between the robot and
its goal, continue moving through open floor without touching it. Never contact a
box through a corner or from a diagonal approach.

## Solve the far-wall box

Prefer the lower-left far-wall box first:

1. Stay behind the boxes and move through open floor into the lower-left lane.
2. Position the robot directly behind the lower-left box, on the near side away
   from the far-wall goal.
3. Verify that the robot, box, and far-wall goal form one straight forward line.
4. Move **forward** to contact the center of the box.
5. Once contact begins, use only **forward**. Do not steer left or right.
6. Inspect after each short push near the goal and stop as soon as the box is
   fully centered inside the far-wall green region.

## Locate the next box

After completing either box, deliberately find the remaining unfinished box
before choosing another push:

1. Move away from the completed box until both the box and its wall are safely
   outside the robot's immediate path.
2. Inspect the scene for the remaining yellow box. An unfinished box is the one
   that is not fully centered on a green region.
3. If it is not visible, move through the open central floor and inspect again
   after each movement. Search from open space; never search by walking along or
   through the completed box.
4. Once the unfinished box is visible, identify its green goal and the open side
   opposite that goal.
5. Move toward that open side without touching the box. Stop with clear space,
   then establish **ROBOT → BOX → GOAL** before making contact.

## Move to the right-wall box

After the far-wall box is complete:

1. Move **backward** until clearly separated from it.
2. Never touch the completed box again.
3. Locate the remaining upper-right box using the procedure above.
4. Return through the open center of the arena without touching the box, then
   move to its left side, opposite the right-wall goal.
5. Match the box's forward depth so the robot, box, and goal form one straight
   horizontal line.
6. Move **right** to contact the center of the box.
7. Once contact begins, use only **right**. Do not move forward or backward.
8. Inspect after each short push near the goal and stop as soon as the box is
   fully centered inside the right-wall green region.

## Completed boxes

A box is complete only when it is fully inside and centered on its green goal.
Once complete:

- Stop pushing immediately.
- Move away in the direction opposite the push: **backward** from the far-wall
  box or **left** from the right-wall box.
- Treat the box as a permanent obstacle.
- Never touch, recenter, or push it again.

## Recovery before completion

If the robot contacts an unfinished box off-center or the box leaves its lane:

1. Stop pushing immediately.
2. Move opposite the push until there is clear space.
3. Route through open floor to the side opposite the goal.
4. Re-establish **ROBOT → BOX → GOAL** through the box centers.
5. Resume only in the box's required direction: **forward** for the far-wall box
   or **right** for the right-wall box.

Never recover or adjust a box that is already centered on its goal.

## Finish

Before issuing **stand**, verify that both yellow boxes are fully inside and
centered on separate green regions and that neither goal is empty. If only one box
is complete, ignore it and continue with the unfinished box.

## Critical rule

**Align the robot before contact, then push each box straight along its existing
box-goal lane without sideways correction.**
