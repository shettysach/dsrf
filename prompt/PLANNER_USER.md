Choose one motion and direction for the robot's next action.

Your entire response must be one line in this exact shape:
{"motion":"walk","direction":"forward"}

The motion value must be stand, walk, or turn.

- walk supports forward, backward, left, or right. Walk left and walk right are
  sideways strafes; they do not change which way the robot faces.
- turn supports only left or right. A turn rotates the robot about a quarter turn
  with little translation, causing the next camera image to face the new direction.
- stand accepts forward, backward, left, or right; the direction is ignored.

All directions are relative to the robot's current facing direction. Use turn when
the robot must face a different direction, especially before walking toward or
pushing something that is not along the current forward axis. Issue one turn at a
time, then inspect the next image and reassess before moving.

Turn-left example:
{"motion":"turn","direction":"left"}

Turn-right example:
{"motion":"turn","direction":"right"}

The first character of your response must be { and the last character must be }.
Never write triple backticks or the word json. Do not include extra fields, comments,
explanations, or any text outside the object.
