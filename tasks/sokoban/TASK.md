# Task: Single-box Sokoban

## Objective

Push the one yellow box onto the one green floor goal. The goal is near one
edge of the square arena. Issue `stand` only once the yellow box visibly
overlaps green.

## Scene and controls

- Dark walls bound the arena.
- The yellow cube is the only movable box; the green square is its only goal.
- Walking into the box pushes it. The box cannot be pulled.
- `forward` and `backward` move along the facing direction. `left` and `right`
  strafe without rotating. `stand` with a direction rotates in place.

## Strategy

Use the current observation to choose each move; the goal edge and push
direction vary between episodes.

1. Identify the yellow box and green goal.
2. Before touching the box, move through clear floor to the side opposite the
   goal. Confirm a straight **robot -> box -> goal** line.
3. Push only while this line is visibly aligned. Make one short push, then
   inspect the resulting observation.
4. After a collision or an off-centre contact, do not repeat the same command
   automatically. Move away, reassess the latest image, and rebuild the line.
5. Stop immediately when the box overlaps green. Do not try to improve a
   completed placement.

## Critical rule

**Choose the push direction from the visible box and goal, not from a fixed
layout or remembered plan.**
