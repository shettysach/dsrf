# Task: seesaw

## Objective

Get onto the blue hinged plank and balance the red counterweight. Finish only
when you are upright with both feet on the plank, the plank is nearly level, and
its motion has settled.

## Scene

- You control a Unitree G1 robot.
- The blue plank is narrow front-to-back and long from image-left to image-right.
- It pivots on the gray support at its center.
- The red counterweight is fixed near one lateral end. Balance from the opposite
  side; never step on or against the red block.

## Approach and mount

Complete these three steps in order. Do not attempt a foot placement until the
ground approach is complete.

1. **Analyze the seesaw.** Identify which end of the blue plank is closest to
   the ground. This is the low end. Do not approach the raised end or the red
   counterweight.
2. **Move to the low end with 2D waypoints only.** Put each waypoint on visible
   floor as close as safely possible to that low end. Never put a waypoint on
   the seesaw itself, a corner, or the counterweight. Stop, face the low end,
   and reassess the image. Use a simple motion command like "walk".
3. **Climb with foot end-effector targets only.** Omit `waypoints_2d` entirely
   and target a foot on the low blue end. Keep the other foot clear, then bring
   it onto the same surface with another foot target. Do not combine a waypoint
   with a foot target. If a foot target is out of range, return to step 2: move
   closer on the floor with a waypoint, stop, then try the foot target again.

Do not use a hand target unless it is needed to recover balance. Do not reach
toward the counterweight.

## Balance

1. Once both feet are on the plank, make only short lateral adjustments along
   its length, away from the red counterweight.
2. After each adjustment, observe the plank and let it settle before deciding
   again. Do not rapidly alternate sides to chase its swing.
3. If your side becomes too low, shift slightly back toward the center. If the
   red-counterweight side remains low, shift slightly farther toward the opposite
   end.
4. Stay near the plank's front-to-back centerline. Use only tiny corrections in
   that direction to keep both feet safely on the surface.
5. While on the plank, use end-effector targets only; do not issue any 2D
   waypoints.

## Finish

Issue a standing motion only after the plank is close to level, no longer
swinging appreciably, and both feet are securely on the blue plank.
