Inspect the current board image and choose exactly one next action. Directions are
image directions: `up` moves toward the top edge, `right` toward the right edge,
`down` toward the bottom edge, and `left` toward the left edge. Each action
attempts exactly one player move. The next image is authoritative; reassess it
before selecting another action.

Respond with raw JSON only. Your first character must be `{` and your last
character must be `}`. Do not use a Markdown code fence or any backticks. Do not
include an explanation, reasoning, commentary, an
array, a second JSON object, or any field besides `action`.

The response has exactly this shape:

{"action":"up"}

The allowed values are `up`, `down`, `left`, `right`, and `reset`. Use `reset`
only when a prior push has made the board unsolvable. Do not reset merely because
the next move is uncertain; make the safest useful one-cell move and inspect the
next board image.
