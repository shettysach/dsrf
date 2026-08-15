# Task: Sokoban

## Objective

Push both yellow boxes fully and separately onto the two green floor goals.
Do not stop after reaching one goal. Issue `stand` only when each green region
contains a yellow box centered inside it.

## Read the board before touching a box

- Yellow cubes are movable boxes. Green floor squares are goals. The dark
  boundary walls are immovable.
- A box can be pushed by walking into it, but cannot be pulled. A bad sideways
  or diagonal push may therefore make the task unsolvable.
- This board is intentionally pre-aligned: one box has a green goal directly
  ahead of it at the far wall, and the other has a green goal directly to its
  right at the right wall. Do not invent a route that moves either box across
  the central open floor.
- The open central floor is for the robot to reposition between pushes; it is
  not a box route.

## Non-negotiable push rule

Before every contact, create this straight arrangement:

**robot -> center of box -> center of its green goal**

Approach only from the side opposite the goal. If the box is not centered in
front of you with its goal visibly beyond it, do not touch it: back away and
reposition through clear floor. Never push a box from a corner, diagonally, or
while turning.

## Explicit solve strategy

Solve the far-wall lane first, then the right-wall lane.

### 1. Far-wall box

1. Use the clear central floor to reach the near side of the box whose goal is
   directly ahead against the far wall. Avoid contact while positioning.
2. Face the box squarely and verify one straight forward line: robot, box,
   then green goal.
3. Walk `forward` to make centered contact. Once contact begins, use only
   `walk forward`; never strafe or turn during this push.
4. As the box reaches the green region, make short forward pushes and inspect
   after each one. Stop immediately when the box is fully inside and centered
   on the green goal.
5. Walk `backward` to create clear separation. The completed box is now a
   permanent obstacle: never touch it again.
6. Then move toward the open center of the arena before searching for the
   remaining yellow box. Do not linger beside the completed box or wall.

### 2. Right-wall box

1. From the open center, locate the remaining yellow box. Approach its left side,
   opposite the green goal at the right wall, without contacting the box.
2. Align at the same forward depth as the box so the robot, box, and goal make
   one horizontal line.
3. Walk `right` to make centered contact. Once contact begins, use only
   `walk right`; do not walk forward, backward, or rotate during this push.
4. Near the goal, use short right pushes with an inspection between them.
   Stop immediately when the box is completely inside and centered on the
   green region.
5. Walk `left` away from the completed box.

## Repositioning and recovery

- `forward`/`backward` are along the current facing direction; `left`/`right`
  are strafes and do not turn the robot. Use `stand` with the desired direction
  to rotate in place when needed to face a safe open route, then inspect again
  before walking.
- Do not use the same command repeatedly without checking the updated image,
  especially near a box or wall.
- If an unfinished box was contacted off-center, stop pushing at once, move
  away opposite the attempted push, route through open floor, and rebuild the
  robot -> box -> goal line. Resume only in that box's original lane.
- When one box turns green, first move away from it and return to the open
  center of the arena. From there, locate and approach the only remaining
  yellow box. Never try to correct a completed box.

## Finish check

Before `stand`, visibly confirm that no green goal is empty and no yellow box
is partly outside its goal. If either condition is not true, continue only with
the unfinished lane.

## Critical rule

**Preserve the two pre-aligned lanes: align before contact, then push each box
straight toward its own goal with no sideways correction.**
