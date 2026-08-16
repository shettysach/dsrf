Choose exactly one next one-cell action from the current board image. Reply with exactly one line of raw JSON. The first character of your response must be `{`. The last character must be `}`. Do not put the JSON in a code fence. Do not write any words before or after it.

Your response must parse as JSON and must contain exactly this one key:

{"action":"up"}

You can use with one of: `up`, `down`, `left`, `right`, or `reset`, for the "action" field values. No other keys, values, arrays, explanations, reasoning, labels, Markdown, or backticks are allowed. An invalid response is rejected and does not move the board.

Use `reset` only after a prior push has made the board unsolvable.
