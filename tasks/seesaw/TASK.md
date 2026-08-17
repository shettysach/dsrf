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

## Required sequence

Follow this sequence exactly. Do not rush to climb or balance.

1. **Go toward the seesaw.** Use only 2D waypoints on clear, visible floor to
   approach the seesaw. Do not place a waypoint on the plank, its support, a
   wall, or the red counterweight.
2. **Stop and wait for it to stabilize.** Once near the seesaw, use a brief
   hold-position motion, then inspect the next observation. Do not choose an end
   while the blue plank is still swinging appreciably. Reserve `motion: "stand"`
   for the final completion only.
3. **Identify the lower blue end.** After it settles, determine which end of the
   blue plank is closest to the floor. This is the only end to approach. Never
   approach the raised end or the red counterweight.
4. **Move to the lower end.** Use 2D floor waypoints only to move close enough
   to climb onto that low end. Stop and reassess before attempting a foot
   placement.
5. **Climb onto the blue region.** Use foot end-effector targets only, with each
   target visibly on the blue plank. First place one foot on the low blue end,
   then bring the other foot onto the blue region. Keep both feet away from the
   red block. Do not combine a waypoint with a foot target. If a foot target is
   out of range, return to step 4.

Do not use a hand target unless it is needed to recover balance. Do not reach
toward the counterweight.

## Balance

1. Once both feet are on the blue plank, wait for it to settle before making an
   adjustment.
2. Make only short lateral adjustments along its length, away from the red
   counterweight. After every adjustment, observe and wait for the plank to
   stabilize again; do not rapidly alternate sides to chase its swing.
3. If your side becomes too low, shift slightly back toward the center. If the
   red-counterweight side remains low, shift slightly farther toward the opposite
   end.
4. Stay near the plank's front-to-back centerline. Use only tiny corrections in
   that direction to keep both feet safely on the blue surface.
5. While on the plank, use end-effector targets only; do not issue any 2D
   waypoints.

## Finish

Issue a standing motion only after the plank is close to level, no longer
swinging appreciably, and both feet are securely on the blue plank.
