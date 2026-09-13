# Task: Palm-assisted Box Push

Run the following scripted sequence:

1. Walk forward to the staging waypoint.
2. Stand and extend both palms toward the near face of the box.
3. Walk forward with both arms held forward, moving the box to the green goal.

ARDY receives timed 2D root-position targets plus sparse bilateral hand
keyframes. The hand keyframes constrain endpoint positions and forward-facing
palm orientation through wrist/hand pose constraints, aiming about 0.65 m
above the floor. A free-moving box starts with
its near face at the nominal palm reach. During push only, a horizontal virtual
force assists the box while a measured palm stays within 5 cm of its near face; it
remains active until the gap exceeds 12 cm. The force stops at the box goal or
when the phase ends. No artificial force is applied to G1.
