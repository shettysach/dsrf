Inspect the current board image and choose exactly one next action. Directions are
image directions: `up` moves toward the top edge, `right` toward the right edge,
`down` toward the bottom edge, and `left` toward the left edge. Each action
attempts exactly one player move. The image after the action is authoritative;
reassess it before choosing the following move.

## Required response schema

Your entire response must be a valid JSON object with exactly one field named
`action`:

```json
{"action":"up"}
```

The only valid values are:

```json
{"action":"up"}
{"action":"down"}
{"action":"left"}
{"action":"right"}
{"action":"reset"}
```

Do not return Markdown, a code fence, explanation, reasoning, commentary, an
array, a second JSON object, or any field besides `action`. In particular,
`{"action":"right","reason":"push box"}` is invalid.

Use `reset` only when a prior push has made the board unsolvable. Do not reset
merely because the next move is uncertain; make the safest useful one-cell move
and inspect the next board image.
