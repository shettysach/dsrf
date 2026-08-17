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

1. **Walk directly in front of the smaller stair.** Identify the smaller block,
   then use a `walk` motion with 2D waypoints on visible floor to stop squarely
   in front of it, as close as possible without contacting it. Do not put a
   waypoint on either block or the platform, and do not approach the larger block.
2. **Squat in place.** Omit `waypoints_2d`, stay facing the smaller block, and
   issue a concise squat motion. Reassess only once the robot is low, balanced,
   and within hand reach.
3. **Grasp and lift it.** Omit `waypoints_2d` entirely and use one or both hand
   end-effector targets on an accessible side of the smaller orange block. Use a
   concise motion such as "grasp and lift the small block". The block attaches
   automatically when either palm reaches it. If it is out of range, stand,
   repeat step 1 to move closer, then squat again before retrying.

## Safety

- Never combine a waypoint with a hand target in one command.
- If using both hands, do not target them at the same image point; use opposite
  accessible sides of the smaller block.
- Make no contact with the larger block, side walls, or platform.

## Finish

Once the smaller block has been lifted clear of the floor, stand while holding
it. Do not continue toward the platform.
