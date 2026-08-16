Choose a motion and zero or more ordered image waypoints on the visible floor.

Your entire response must be one line in this exact shape:
{"motion":"walk","waypoints_2d":[[350,700],[650,600]]}

The motion value must be stand or walk. waypoints_2d is an ordered list of
zero or more [x,y] points. Coordinates are integers in [0,1000], where [0,0]
is the top-left image corner and [1000,1000] is the bottom-right. Select only
visible floor points. Use [] when no image waypoint is needed.

The first character of your response must be { and the last character must be }.
Never write triple backticks or the word json. Do not include extra fields, comments,
explanations, or any text outside the object.
