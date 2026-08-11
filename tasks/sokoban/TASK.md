# Task: Sokoban

## Objective

Push both yellow boxes completely onto the two green goal regions.

The left box belongs on the left goal and the right box belongs on the right goal.
Each goal is directly ahead of its box. When both boxes are centered on their goals,
issue **stand**.

## Scene and controls

- The elevated camera looks forward over the humanoid.
- Yellow cubes are movable boxes.
- Green floor squares are goal regions.
- Dark walls bound the arena.
- Boxes are very light. Walking into a box pushes it in your direction of travel.
- You can push boxes, but you cannot pull them.
- **forward** and **backward** move along the robot's facing direction.
- **left** and **right** move laterally; they do not turn the robot around.

## Critical alignment rule

**Align the robot before touching a box.**

For a correct forward push, the center of the robot, the center of the chosen box,
and the center of that box's green goal must form one straight line in the image.
The robot must be directly behind the box, not behind one of its corners.

Do not begin pushing merely because the robot is close to a box. First use **left**
or **right** while still behind the boxes until the robot is centered in the chosen
box's lane. Only then use **forward**.

## Procedure

Complete one box before working on the other:

1. Choose either the left or right box.
2. Stay behind the boxes and move laterally into the chosen box's lane.
3. Check alignment before contact:
   - the robot is centered behind the box;
   - the robot is facing squarely toward the box;
   - the goal is centered directly beyond the box;
   - the intended path is clear.
4. If any part is misaligned, keep repositioning. Do not push yet.
5. Once aligned, walk **forward** into the center of the box.
6. Continue with straight forward pushes while the box remains centered on its goal.
7. Near the goal, use short forward pushes and check the image after every push.
8. Stop pushing when the entire box is inside the green square and centered.
9. Walk **backward** far enough to clear the placed box.
10. Move laterally behind the remaining box, align again, and repeat.

## While pushing

- Push through the center of the box, never through a corner.
- Do not issue **left** or **right** while touching a box that should move forward.
- Do not approach diagonally.
- Do not push a box past its green goal.
- Do not touch or dislodge a box that is already correctly placed.
- Keep boxes away from the perimeter walls; a box against a wall may be impossible
  to recover because it cannot be pulled.

## Recovery

If a box drifts sideways or the robot contacts it off-center:

1. Stop the forward push immediately.
2. Walk **backward** until there is clear space between the robot and box.
3. Use **left** or **right** to center the robot behind the box again.
4. Verify the robot–box–goal centerline.
5. Resume with one short **forward** push.

Never try to fix bad alignment by continuing to push from the wrong angle.

## Finish

Before issuing **stand**, verify that both yellow boxes are fully inside separate green
goal regions and neither green region is empty. A box touching only the edge of a
goal is not complete.
