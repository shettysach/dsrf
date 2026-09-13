# Task: Contact-free Push Motion

There is no box to touch or move. Run the following scripted sequence:

1. Walk forward to the staging waypoint.
2. Stand and extend both arms straight forward.
3. Walk forward with both arms held forward to the goal waypoint.

ARDY receives timed 2D root-position targets plus sparse bilateral hand-position
keyframes. The script describes forward-facing palms, but the timed path does
not currently enforce palm orientation. There are no wrist-joint, torso,
upright-root, contact, or other pose constraints.
