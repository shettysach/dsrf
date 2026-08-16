# Task: Single-box Sokoban

Push the one yellow box onto the one green floor goal near the arena edge.
Issue `stand` only after the yellow box visibly overlaps green.

- The yellow cube can be pushed by walking into it but cannot be pulled.
- `forward`/`backward` move in the facing direction; `left`/`right` strafe;
  `stand` rotates in place.
- Before contact, use open floor to establish a straight **robot -> box -> goal**
  line. Push only along that visible line.
- Inspect after every contact or push. After a collision or off-centre contact,
  move away and reassess; never repeat the same command just because it was part
  of an earlier plan.
- Do not move the box once it visibly overlaps the green goal.
