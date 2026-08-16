import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const MIN_WINDOW = 4;
const MAX_WINDOW = 8;
const DEFAULT_WINDOW = 6;
const DEBUG_ENABLED = !["", "0", "false", "no", "off"].includes(
  (process.env.PI_DEBUG ?? "").trim().toLowerCase(),
);

/**
 * Keep a short, literal interaction history for the VLM.  A turn begins with
 * an external observation (a user message) and includes the associated action
 * and tool result.  Pi still saves the full session for debugging.
 */
export default function slidingContext(pi: ExtensionAPI): void {
  const windowSize = contextWindowSize();

  pi.on("session_start", async () => {
    pi.setActiveTools(["robot_action"]);
  });

  pi.on("context", async (event) => {
    const firstRetained = observationStart(event.messages, windowSize);
    const messages = event.messages.slice(firstRetained);
    debug(
      `retaining ${messages.length}/${event.messages.length} messages ` +
        `for the latest ${countObservations(messages)} observation turns ` +
        `(window=${windowSize})`,
    );
    return { messages };
  });
}

function contextWindowSize(): number {
  const value = process.env.PI_CONTEXT_WINDOW?.trim();
  if (!value) return DEFAULT_WINDOW;

  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < MIN_WINDOW || parsed > MAX_WINDOW) {
    throw new Error(
      `PI_CONTEXT_WINDOW must be an integer from ${MIN_WINDOW} to ${MAX_WINDOW}; got ${value}`,
    );
  }
  return parsed;
}

function observationStart(
  messages: ReadonlyArray<{ role: string }>,
  windowSize: number,
): number {
  const observationIndexes: number[] = [];
  for (const [index, message] of messages.entries()) {
    if (message.role === "user") observationIndexes.push(index);
  }
  return observationIndexes.at(-windowSize) ?? 0;
}

function countObservations(messages: ReadonlyArray<{ role: string }>): number {
  return messages.filter(({ role }) => role === "user").length;
}

function debug(message: string): void {
  if (DEBUG_ENABLED) console.error(`[sliding-context] ${message}`);
}
