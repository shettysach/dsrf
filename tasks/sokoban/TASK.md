# Task: Sokoban

## Objective

Push both yellow boxes completely onto the two green goal regions.

The task is complete only when each green region contains one box. Once both boxes
are centered on goals, issue **stand** and remain standing.

## Scene

- You control the humanoid shown near the bottom of the elevated camera view.
- The two yellow cubes are movable boxes.
- The two green floor squares are goal regions. They do not block movement.
- The dark perimeter walls bound the arena.

## Box mechanics

- Boxes are very light and slide along the floor.
- Push a box by walking directly into it; there is no special push or grasp action.
- A box moves away from you in the direction you walk into it.
- You can push boxes, but you cannot pull them.
- To change a box's direction, stop, move around it, align from a new side, and push
  again.

## Plan before pushing

1. Locate both boxes, both goals, the robot, and the arena walls.
2. Choose a reachable goal for each box.
3. Preserve enough open space to move behind every box that still needs pushing.
4. Avoid pushing a box against a perimeter wall. A box at a wall may be impossible
   to recover because you cannot get behind it.
5. Work on one box at a time unless repositioning requires otherwise.

## How to make a controlled push

1. Select the box and goal you are currently working on.
2. Move to the side of the box opposite the goal.
3. Align the robot, box center, and goal center into one straight line.
4. Face the box squarely before making contact.
5. Walk straight into the center of the box.
6. Use short pushes near the goal so you do not overshoot.
7. Stop when the box is fully inside and centered on the green region.

Do not push diagonally when a straight, centered push is possible. Contact near a
corner can send the box off course.

## Recovery

If a box moves off the intended line:

1. Stop pushing immediately.
2. Back away far enough to walk around the box without touching it.
3. Move to the correct side.
4. Realign the robot, box, and goal.
5. Resume with a short straight push.

Do not repeatedly push from a bad angle. Do not disturb a box that is already
centered on a goal.

## Finish

Before finishing, verify that:

1. Both yellow boxes are visibly inside green goal regions.
2. No green goal is empty.
3. The boxes are centered rather than merely touching a goal edge.

Then issue **stand**.
