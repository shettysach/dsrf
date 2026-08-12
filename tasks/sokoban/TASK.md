# Task: Sokoban

## Objective

Push both yellow boxes completely onto the two green goal regions.

The goals are against different walls and require different push directions:

- Push the lower-left box **forward** onto the green region at the far wall.
- Push the upper-right box **right** onto the green region at the right wall.

When both boxes are centered on their goals, issue **stand**.

## Placed boxes are finished

**Once a box is fully inside and centered on a green goal, never push or touch that
box again.** Treat it as locked in place and permanently complete.

After placing a box:

1. Stop immediately.
2. Move opposite the completed push to create clear separation: **backward** from
   the far-wall box or **left** from the right-wall box.
3. Route around it without making contact.
4. Work only on the remaining unplaced box.

Do not use a placed box for alignment, do not walk through it, and do not make an
extra "small correction" after it is already centered. Moving a completed box off
its green region undoes progress.

## Scene and controls

- The elevated camera looks forward over the humanoid.
- Yellow cubes are movable boxes.
- Green floor squares are goal regions.
- Dark walls bound the wide arena.
- One green goal touches the far wall; the other touches the right wall at
  approximately the middle depth of the arena.
- Boxes are very light. Walking into a box pushes it in your direction of travel.
- You can push boxes, but you cannot pull them.
- **forward** and **backward** move along the robot's facing direction.
- **left** and **right** move laterally; they do not turn the robot around and can
  be used to push a box sideways.

## Critical alignment rule

**Align the robot before touching a box.**

For any push, the center of the robot, the center of the chosen box, and the center
of that box's green goal must form one straight line in the intended push direction.

- For the far-wall box, stand directly behind it and push **forward**.
- For the right-wall box, stand directly to its left at the same depth and push
  **right**.

Never push through a corner.

Do not begin pushing merely because the robot is close to a box. First move through
open floor until the robot is on the side opposite that box's goal and centered on
the box-goal line. Only then move toward the box.

## Procedure

Complete one box before working on the other. Prefer the far-wall box first:

1. Stay behind both boxes and move **left** into the lower-left box's lane.
2. Verify that the robot, box, and far-wall goal form a straight forward line.
3. Push only **forward**, checking the image after each command near the goal.
4. Stop when the entire box is centered inside the far-wall green region.
5. Walk **backward** until clearly separated, then return through the open center.
6. Approach the upper-right box without touching it and move to its left side.
7. Match the box's forward depth so the robot, box, and right-wall goal form a
   straight horizontal line in the image.
8. Push only **right**, checking the image after each command near the goal.
9. Stop when the entire box is centered inside the right-wall green region.

## While pushing

- Push through the center of the box, never through a corner.
- Use **forward** only for the far-wall box and **right** only for the right-wall box
  once contact begins.
- Do not approach or push diagonally.
- Do not push a box past its green goal.
- A box correctly placed on green is finished: never touch, push, or correct it
  again.
- Keep boxes away from every wall except their intended wall-touching goal. A box
  against the wrong wall may be impossible to recover because it cannot be pulled.

## Recovery

If a box drifts off its box-goal line or the robot contacts it off-center:

1. Stop pushing immediately.
2. Move opposite the failed push until there is clear space: **backward** after a
   forward push or **left** after a right push.
3. Return to the side opposite the goal: behind the far-wall box or left of the
   right-wall box.
4. Verify the robot–box–goal centerline.
5. Resume with one short push in the correct direction: **forward** or **right**.

Never try to fix bad alignment by continuing to push from the wrong angle.
This recovery procedure applies only to a box that is not yet correctly centered on
a goal. Never recover, adjust, or re-push a completed box.

## Finish

Before issuing **stand**, verify that both yellow boxes are fully inside separate green
goal regions and neither green region is empty. A box touching only the edge of a
goal is not complete.

If one box is complete and the other is not, ignore the completed box entirely and
act only on the unfinished box.
