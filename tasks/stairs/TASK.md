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

Complete the following phases in order.

1. **First, identify the smaller block.** Do not approach the larger block.
2. **Then, approach with 2D waypoints only.** Place waypoints on visible floor,
   stopping directly in front of the smaller block and close enough for both
   hands to reach it. Do not put a waypoint on either block or the platform.
3. Stop, face the smaller block squarely, and reassess the image. Keep the block
   centered in front of the robot.
4. **Then, pick up with hand end-effector targets only.** Omit `waypoints_2d`
   entirely. Place `left_hand` and `right_hand` targets on the two accessible
   sides of the smaller orange block, not on the floor and not on the larger
   block. Use a concise motion such as "grasp and lift the small block".
5. Keep both hand targets within reach. If either hand target is too far, return
   to the ground-approach phase, walk closer, stop, and try again.

## Safety

- Never combine a waypoint with a hand target in one command.
- Do not target both hands at the same image point; target opposite sides of the
  smaller block.
- Make no contact with the larger block, side walls, or platform.

## Finish

Once both hands are on the smaller block and it has been lifted clear of the
floor, stand while holding it. Do not continue toward the platform.
