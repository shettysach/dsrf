# Task: stairs

## Objective

Find the smaller orange stair block, approach it, and pick it up with both
hands. This is the only objective: do not move either block toward the raised
green platform and do not attempt to climb the stairs.

## Scene

- Two orange movable stair blocks sit on the floor. The smaller block is shorter,
  narrower, and lower than the larger block.
- A raised gray platform with a green top is in the back of the arena. Ignore it.
- The camera follows the robot's position but keeps a fixed world-facing view.

## Procedure

Complete these three steps in order.

1. **Move to the smaller stair.** Identify the smaller block, then use 2D
   waypoints on visible floor to stop directly in front of it. Do not put a
   waypoint on either block or the platform. Do not approach the larger block.
2. **Squat in front of it.** Omit all image constraints, face the smaller block
   squarely, and issue a concise squat motion. Reassess once the robot is low,
   balanced, and close enough to reach the block with both hands.
3. **Pick it up with both hands.** Omit `waypoints_2d` entirely and use
   `left_hand` and `right_hand` end-effector targets on opposite accessible sides
   of the smaller orange block. Use a concise motion such as "grasp and lift the
   small block". The block attaches automatically when either palm reaches it.
   If either target is out of range, repeat step 1 to move closer, then squat
   again before retrying.

## Safety

- Never combine a waypoint with a hand target in one command.
- Do not target both hands at the same image point; target opposite sides of the
  smaller block.
- Make no contact with the larger block, side walls, or platform.

## Finish

Once both hands are on the smaller block and it has been lifted clear of the
floor, stand while holding it. Do not continue toward the platform.
