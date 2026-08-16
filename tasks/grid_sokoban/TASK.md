# Task: 5×5 Grid Sokoban

## Objective

Put every yellow box onto a green goal cell. The task completes automatically
when every box covers a goal.

## Visual legend

- A dark gray cell is a wall. It cannot be entered and no box can enter it.
- A pale cell is empty floor.
- A solid green square filling an entire cell is a goal.
- A yellow square with a dark outline is a movable box.
- A blue circle is the player.
- A yellow box drawn over green means that box is already on its goal.

The playable board is exactly 5 cells wide by 5 cells tall, inside a one-cell
border of walls. Read directions directly from the image: up is toward the top,
right is toward the right, down is toward the bottom, and left is toward the
left.

## Rules

Each response attempts to move the blue player by exactly one cell. The player
can walk onto empty floor or an uncovered goal. Walking into a box pushes that
box one cell in the same direction only when the cell beyond the box is empty
floor or a goal. A box cannot be pulled. A move into a wall or into a box whose
far side is blocked has no effect.

## Planning procedure

1. Locate every green goal, yellow box, wall, and the blue player before moving.
2. For each box, identify the goal it can reach and the direction of its final
   push.
3. Check that the player can reach the cell on the opposite side of the box
   before making that push.
4. Avoid pushing a box into a non-goal corner or against a wall where it can no
   longer reach its goal.
5. Make one move, inspect the new image, and update the plan. Do not assume a
   blocked move changed the board.
6. Use `reset` only after the board has become unsolvable; it restores the
   original layout.
