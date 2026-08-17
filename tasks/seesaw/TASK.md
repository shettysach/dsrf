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

Complete these phases in order. Do not attempt a foot placement on the seesaw
until the ground approach is complete.

1. **First, approach with 2D waypoints only.** Put each waypoint on the visible
   ground directly in front of the center of the near long edge of the blue
   plank. Do not put a waypoint on the plank, a corner, or either end.
2. Stop at that ground position, face the plank, and reassess the image.
3. **Then, mount with foot end-effector targets only.** Target a foot on the blue
   surface near the hinge line and its front-to-back centerline. Keep the other
   foot clear, then bring it onto the same central area with another foot target.
4. Do not use a hand target unless it is needed to recover balance. Do not reach
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

## Finish

Issue a standing motion only after the plank is close to level, no longer
swinging appreciably, and both feet are securely on the blue plank.
