Choose a motion and optionally ground either navigation waypoints or end-effector
targets in the image.

For navigation, use this shape:
{"motion":"walk","waypoints_2d":[[350,700],[650,600]]}

For a hand or foot target, use this shape:
{"motion":"reach toward the red box","end_effectors":[{"name":"right_hand","target_2d":[600,400]}]}

motion is a concise natural-language motion description. waypoints_2d is an
ordered list of visible floor points. end_effectors is a list with at most one
entry per end effector; name must be left_hand, right_hand, left_foot, or
right_foot. target_2d must select the visible object surface the hand should
reach, or the floor surface a foot should step toward. Do not combine
waypoints_2d and end_effectors in one command.

All image coordinates are integers in [0,1000], where [0,0] is the top-left and
[1000,1000] is the bottom-right. Omit both optional fields when image grounding
is unnecessary; this avoids requesting a depth map.

The first character of your response must be { and the last character must be }.
Never write triple backticks or the word json. Do not include extra fields,
comments, explanations, or any text outside the object.
