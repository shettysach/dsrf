# Task: Sokoban

## Objective

Push every yellow box completely onto a separate green goal square. The board is
an 8×8 Sokoban grid rendered in 3D: every dark wall block, box, and goal occupies
one grid cell. Some levels have two boxes and some have three.

## Rules

- Boxes can be pushed but never pulled.
- Walls and other boxes block movement.
- Before contact, align the robot, box centre, and intended destination on one
  straight grid lane.
- Use short, straight pushes near a goal. A box is finished only when it is fully
  inside the green square.
- Do not move a box that is already correctly placed on a goal.
- Plan around internal walls and avoid pushing a box into a corner unless that
  corner is its goal.

## Finish

Issue **stand** only after every goal contains one box and no box is off a goal.
