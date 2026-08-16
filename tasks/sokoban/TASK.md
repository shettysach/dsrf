# Task: Single-box Sokoban

## Objective

Push the one yellow box onto the one green floor goal near the arena edge.
Issue `stand` only after the yellow box visibly overlaps the green square.

## Scene and controls

- The arena is open and square. Dark walls are immovable.
- The yellow cube is the only movable box. The green square is its only goal.
- Walking into the box pushes it; the box cannot be pulled.
- `forward` and `backward` move along the current facing direction. `left` and
  `right` strafe without rotating. `stand` with a direction rotates in place.

## Control policy

Use the current RGB image and the recent observation history to make one small,
visually justified move.

1. Find the yellow box and green goal. Work out the straight direction from the
   box toward the goal.
2. Before touching the box, move through open floor to the side opposite the
   goal. Establish a straight **robot -> box -> goal** line.
3. Push only when that line is visibly aligned. Use one short walk at a time and
   inspect the next observation after every contact or push.
4. If a command collides or the box is no longer aligned with the goal, do not
   repeat that command by habit. Move away, reassess the new image, and restore
   the robot -> box -> goal line.
5. Stop immediately once the yellow box visibly overlaps green. Do not try to
   center or improve a completed placement.

## Critical rule

**Do not choose a direction from a remembered plan alone. Verify the box, goal,
and clear approach in the latest RGB observation before every push.**
