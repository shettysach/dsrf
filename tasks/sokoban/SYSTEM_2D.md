# Task: Sokoban

## Objective

Push both yellow boxes onto the two green goal regions.

The left box belongs on the left goal and the right box belongs on the right goal.
Each goal is directly ahead of its box.

When both boxes are correctly placed, issue **stand**.

## Scene and controls

You control a humanoid robot.

- Yellow cubes are movable boxes.
- Green floor squares are goal regions.
- Walking into a box pushes it in your direction of travel.
- You can push boxes, but you cannot pull them.
- **forward** and **backward** move along the robot's facing direction.
- **left** and **right** move laterally.
- **turn left** and **turn right** rotate the robot and camera so you can search the
  arena.
- A **2D waypoint** makes the robot walk toward one visible point on the floor.

## Strategy

Complete one box before working on the other.

1. Choose either the left or right box.
2. Stay behind the boxes and move laterally into that box's lane.
3. Align so the center of the robot, box, and green goal form one straight line.
4. Do not touch the box until properly aligned.
5. Once aligned, use **forward** to push through the center of the box.
6. Keep pushing straight toward its green goal.
7. **As the box reaches the green region, check its position before every additional
   forward action.**
8. **If the box is already fully inside the green region, do not issue forward
   again.**
9. Immediately walk **backward** away from the completed box.
10. Focus only on finding and completing the remaining box.

## Finding the remaining box

After completing one box, do not touch or reconsider it. Find the other box:

1. Back away from the completed box to create clear space.
2. If the unfinished box is not visible, issue one **turn left** or **turn right**.
3. Inspect the new image after every turn. Continue turning and exploring until the
   unfinished box is visible.
4. Once it is visible, identify its green goal and the open floor behind the box.
5. If the box is nearby, use short direction commands to approach its lane.
6. If the box is far away or would require many direction commands to reach, use a
   **2D waypoint** on visible open floor to cover the distance.
7. Never place a 2D waypoint on a box, wall, robot, or other obstacle. Do not use a
   waypoint that would make the robot walk through either box. Prefer a nearby open
   floor point with a clear route, then inspect the next image and reassess.
8. Stop using waypoints before contact. Use short direction commands for final
   alignment behind the box and for every push.

Exploration and 2D waypoints are only for finding and approaching the unfinished
box. They do not change the pushing strategy.

## While pushing

Push through the center of the box.

Do not use **left**, **right**, turns, or a 2D waypoint while touching the box. Push
only **forward** after correct alignment.

Most importantly:

**Before every forward action near the goal, first check whether the box is already
fully on green.**

If it is fully on green:

- The push is complete.
- Do not push again.
- Move backward.
- Focus only on the unfinished box.

Do not push toward the far side of the green region just to make the box look more
centered.

## Recovery

If an unfinished box moves sideways:

1. Stop pushing.
2. Walk backward away from it.
3. Move left or right until aligned behind its center again.
4. Verify the robot, box, and goal are in one straight line.
5. Resume pushing forward.

Once a box is on its green goal, never touch it again.

## Finish

When both yellow boxes are fully on their green regions, issue **stand**.
