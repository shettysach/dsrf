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

The `TASK STATE` block is the only persistent task memory. It contains a small
subtask plan, not a scene description.

On the initial observation, assign `box_1` and `box_2` using one stable visible
convention, such as left-to-right. Never swap these labels later. Initialize
these two subtasks with concise, visually grounded objectives:

```text
place_box_1: place box_1 onto its matched green goal
place_box_2: place box_2 onto its matched green goal
```

Every observation must make exactly two tool calls in one response: first
`update_state`, then `robot_action`. State has only:

- `active_subtask`: the one subtask currently being worked on.
- `subtasks`: updates to `place_box_1` or `place_box_2`, each with status
  `pending`, `active`, `complete`, or `blocked` and its stable objective.
- `subgoal`: one immediate visual action objective.

Do not store reasoning, scene narration, robot position, directions, collision
summaries, or an action transcript. The current observation already supplies
the completed command and collision outcome. A completed subtask is permanent:
never reopen it or approach that completed box again.

When the first subtask becomes complete, set the other subtask active and make
the next subgoal specifically reach the goal-opposite side of its box through
visible clear floor. Do not alternate `forward` and `backward` as a generic
recovery. For a right-side goal, move to the left side of the active box and
use `walk right` to push; `left` and `right` already strafe without rotating.

The `robot_action` call ends the Pi turn; wait for the resulting observation
before reasoning or acting again.

## Finish

Before issuing `stand`, visibly confirm that no green goal is empty and no
yellow box is partly outside its goal.

## Critical rule

**Update factual task state at each meaningful transition; align before contact,
then push each box straight along its own lane.**
