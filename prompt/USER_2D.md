Choose exactly one of these two command types for the robot's next action:

1. DIRECTION: move or turn relative to where the robot is currently facing.
2. 2D WAYPOINT: walk to one visible point on the floor in the current image.

Use only one type per response. Never combine direction and waypoints_2d.

DIRECTION COMMANDS

Walk forward, backward, left, or right:
{"motion":"walk","direction":"forward"}
{"motion":"walk","direction":"left"}

Walk left and walk right are sideways steps. They do not rotate the robot.

Turn left or right by about a quarter turn:
{"motion":"turn","direction":"left"}
{"motion":"turn","direction":"right"}

Turning changes where the next camera image faces. All directions are relative to
the robot's current facing direction.

2D WAYPOINT COMMAND

Walk to a visible floor point:
{"motion":"walk","waypoints_2d":[[500,700]]}

waypoints_2d is a list of image-coordinate pairs. Each pair contains two integer
coordinates in [0,1000]:

- [0,0] is the top-left corner.
- [1000,0] is the top-right corner.
- [0,1000] is the bottom-left corner.
- [1000,1000] is the bottom-right corner.
- [500,500] is the image center.

The waypoint must be on visible traversable floor, not on a wall, box, or robot.
An unobstructed green region is floor. Prefer a nearby reachable floor point. Use a
2D waypoint when a precise position in the image is more useful than one fixed
direction. Use a direction command for pushing straight, stepping sideways, moving
backward, or turning.

To remain still:
{"motion":"stand","direction":"forward"}

Your entire response must be exactly one JSON object on one line. The first
character must be { and the last character must be }. Do not write markdown,
explanations, comments, or extra fields.
