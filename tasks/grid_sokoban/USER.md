Choose exactly one next one-cell action from the current board image.

Prefer raw JSON on one line:

{"action":"up"}

The `action` value must be one of `up`, `down`, `left`, `right`, or `reset`. Do
not add fields, prose, reasoning, labels, arrays, or a second JSON object.

Raw JSON is preferred. For compatibility, one outer JSON Markdown fence is also
accepted, but it must contain only the JSON object and nothing else. Use `reset`
only after a prior push has made the board unsolvable.
