# Task: stairs

## Objective

Find the smaller orange stair block and approach it safely. This is the only
objective: do not move either block toward the raised green platform and do not
attempt to climb the stairs.

## Scene

- Two orange fixed stair blocks sit on the floor. The smaller block is shorter,
  narrower, and lower than the larger block.
- A raised gray platform with a green top is in the back of the arena. Ignore it.
- The camera follows the robot's position but keeps a fixed world-facing view.

## Procedure

Complete these two steps in order.

1. **Move to the smaller stair.** Identify the smaller block, then use 2D
   waypoints on visible floor to stop directly in front of it. Do not put a
   waypoint on either block or the platform. Do not approach the larger block.
2. **Stand in front of it.** Omit all image constraints and issue a concise
   stand motion once the robot is balanced directly in front of the smaller
   block. Do not touch either block.

## Safety

- Make no contact with the larger block, side walls, or platform.

## Finish

Once standing safely in front of the smaller block, stop. Do not continue toward
the platform.
