# Task: Sokoban with task state

## Objective

Push both yellow boxes fully and separately onto the two green floor goals.
Do not stop after one goal is occupied. Issue `stand` only when both boxes are
centered inside separate green regions.

## Scene and controls

- Yellow cubes are movable boxes; green floor squares are goals; dark walls are
  immovable.
- A box can be pushed but cannot be pulled. Never make a sideways or diagonal
  push that takes a box out of its pre-aligned lane.
- `forward` and `backward` move along the current facing direction. `left` and
  `right` strafe without rotating. `stand` with a direction rotates in place.
- Before contact, establish one straight line: **robot -> box -> goal**. Move
  through open floor to the side opposite the goal if that line is not visible.

## Solve strategy

1. First solve the box whose green goal is directly ahead at the far wall. Once
   centered on green, move backward away from it and never touch it again.
2. Return through the open center and solve the remaining box whose goal is to
   its right at the right wall. Approach from its left side, then push only
   right until it is centered on green.
3. Inspect after every push near a box or goal. If contact is off-center, move
   away, route through clear floor, and rebuild the robot -> box -> goal line.

## Working-memory protocol

The `TASK STATE` block is the only persistent task memory. Keep it factual and
small. On the initial observation, call `update_state` before `robot_action` to
record the first subgoal and the two boxes' progress. Thereafter, call
`update_state` before `robot_action` whenever the subgoal, box completion, an
important alignment fact, or the previous result changes.

Use these fields only:

- `subgoal`: the immediate actionable objective, such as `align behind far box`.
- `progress`: one entry for each box, with status `unknown`, `active`, or
  `complete`.
- `known`: up to five verified visual facts, not guesses.
- `last_result`: the observed outcome of the last completed action.

Do not store reasoning, alternatives, or a movement transcript. If the state
already matches the current observation, do not update it just to repeat it.
After an `update_state` call, call `robot_action` for exactly one safe next
motion.

## Finish

Before issuing `stand`, visibly confirm that no green goal is empty and no
yellow box is partly outside its goal.

## Critical rule

**Update factual task state at each meaningful transition; align before contact,
then push each box straight along its own lane.**
