# Task: Single-box Sokoban

Push the one yellow box onto the one green floor goal near an arena edge. Issue
`stand` only once the box visibly overlaps green.

- The box can be pushed but cannot be pulled.
- Before contact, establish a straight **robot -> box -> goal** line from the
  current image. Approach from the side opposite the goal.
- Use direction commands for every push. Use a 2D waypoint only through visible
  clear floor while repositioning, never onto or through the box.
- Inspect after every contact or push. After a collision or off-centre contact,
  move away and reassess instead of repeating the previous command.
- The goal edge and required push direction vary by episode; rely on the RGB
  observation rather than a memorized layout.
