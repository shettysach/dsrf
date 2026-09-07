# Task: Solve the Sokoban Board

You control the humanoid robot in an 8×8 Sokoban board rendered with physical
3D objects. Push every box onto a goal. There are two or three boxes depending
on the level.

## Visual legend

- **Humanoid robot:** the player. The elevated camera translates with the
  robot, but its orientation is fixed to the board. Screen directions therefore
  remain stable even when the robot turns.
- **Yellow cube:** a movable box not yet on a goal. It is smaller than a grid
  cell, so use its centre rather than an edge as the alignment reference.
- **Bright green floor square:** an unoccupied goal.
- **Dark green cube:** a box whose centre is within a goal square. It
  is complete; never touch or move it again.
- **Open floor:** traversable space between grid cells.
- **Dark solid blocks:** walls and the board boundary. They cannot be crossed.

## Movement and pushing rules

Boxes can be pushed but never pulled. A physical push is continuous, rather
than a discrete one-cell action, so make conservative, short motions and look
again after each command.

Before making contact, place the robot directly behind the selected box. The
robot centre, box centre, and intended next grid-cell centre must form one
straight lane. Push through the middle of the box, not a corner. Do not walk
sideways or diagonally while touching it.

Walls and other boxes block movement. Do not push a box against a wall or into
a non-goal corner: it may be impossible to recover because it cannot be pulled.
Work on one uncompleted box at a time whenever this avoids blocking the route
to another box.

A box is complete once its centre is within a bright green goal square. It
then appears dark green. Leave every dark green box alone permanently, including
while travelling to another box.

## Response policy

Use the provided kinematic-planner tool for every response. Choose one safe,
deliberate command from the current image; do not describe an action in ordinary
text.

The final user image in the conversation is the current board state. An earlier
initial-board image may be retained in the conversation only as context; do not
use it to infer the current positions of the robot or boxes.

- Use `walk` with one relative direction (`forward`, `backward`, `left`, or
  `right`) for a short, controlled reposition or push.
- Use `walk` with image waypoints only when a precise free-floor destination is
  clearer than a relative direction. Do not combine a waypoint and direction.
- Use `stand` only after every box is dark green and every goal is occupied.

After each push, reassess the board image before choosing the next action.
