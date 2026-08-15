Choose a motion and zero or more ordered image waypoints on the visible floor.

The motion value must be stand or walk. waypoints_2d is an ordered list of
zero or more [x,y] points. Coordinates are integers in [0,1000], where [0,0]
is the top-left image corner and [1000,1000] is the bottom-right. Select only
visible floor points. Use [] when no image waypoint is needed.
Call robot_action with the motion and waypoints_2d; do not return a text command.
