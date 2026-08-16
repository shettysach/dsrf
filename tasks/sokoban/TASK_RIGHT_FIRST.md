# Task: Sokoban

## Objective

Push both yellow boxes completely onto their matching green floor goals, one box
at a time. When both boxes are fully centered on separate goals, issue **stand**.

Try the **upper-right box first**. It is already aligned with the green goal
against the right wall and must move only **right**. After completing it, solve
the lower-left box by pushing it only **forward** toward the far-wall goal.

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

## Solve the right-wall box first

Prefer the upper-right box first:

1. Move through open floor toward the upper-right box without touching it.
2. Move to the box's left side, opposite the green goal against the right wall.
3. Match the box's forward depth so the robot, box, and right-wall goal form one
   straight horizontal line.
4. Move **right** to contact the center of the box.
5. Once contact begins, use only **right**. Do not move forward or backward.
6. Inspect after each short push near the goal and stop as soon as the box is
   fully centered inside the right-wall green region.

## Move to the far-wall box

After the right-wall box is complete:

1. Move **left** until clearly separated from it.
2. Never touch the completed box again.
3. Look for the remaining lower-left yellow box. If it is not visible, return
   toward the open center and inspect again.
4. Cross the open center toward the lower-left box without touching either box.
5. Move directly behind the unfinished box, on the near side opposite the
   far-wall goal.
6. Verify that the robot, box, and far-wall goal form one straight forward line.
7. Move **forward** to contact the center of the box.
8. Once contact begins, use only **forward**. Do not steer left or right.
9. Inspect after each short push near the goal and stop as soon as the box is
   fully centered inside the far-wall green region.

## Completed boxes

A box is complete only when it is fully inside and centered on its green goal.
Once complete:

- Stop pushing immediately.
- Move away in the direction opposite the push: **left** from the right-wall box
  or **backward** from the far-wall box.
- Treat the box as a permanent obstacle.
- Never touch, recenter, or push it again.

## Recovery before completion

If the robot contacts an unfinished box off-center or the box leaves its lane:

1. Stop pushing immediately.
2. Move opposite the push until there is clear space.
3. Route through open floor to the side opposite the goal.
4. Re-establish **ROBOT → BOX → GOAL** through the box centers.
5. Resume only in the box's required direction: **right** for the right-wall box
   or **forward** for the far-wall box.

Never recover or adjust a box that is already centered on its goal.

## Finish

Before issuing **stand**, verify that both yellow boxes are fully inside and
centered on separate green regions and that neither goal is empty. If only one box
is complete, ignore it and continue with the unfinished box.

## Critical rule

**Solve the right-wall box first, align before contact, and push each box only
along its existing box-goal lane.**
