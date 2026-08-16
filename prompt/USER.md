Choose a motion and, only when navigation needs it, zero or more ordered image
waypoints on the visible floor.

Your entire response must be one line in this exact shape:
{"motion":"walk","waypoints_2d":[[350,700],[650,600]]}

The motion value must be stand or walk. waypoints_2d is optional. When present,
it is an ordered list of [x,y] points. Coordinates are integers in [0,1000],
where [0,0] is the top-left image corner and [1000,1000] is the bottom-right.
Select only visible floor points. Omit waypoints_2d when no image waypoint is
needed; this avoids requesting a depth map.

The first character of your response must be { and the last character must be }.
Never write triple backticks or the word json. Do not include extra fields, comments,
explanations, or any text outside the object.
