# Task: Grid Sokoban

Solve the top-down Sokoban board shown in the image. Dark gray squares are walls,
the blue circle is the player, yellow squares are boxes, and green circles are goals.
A yellow box covering green is a box correctly placed on a goal.

Move exactly one tile per response. Walking into a box pushes it one tile only if
the space beyond it is open. Boxes cannot be pulled. Plan before pushing: never
push a box into a non-goal corner, and make sure the player can reach the required
side of every box. The episode ends automatically once every box is on a goal.
