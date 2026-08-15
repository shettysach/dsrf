import {
  isToolCallEventType,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import { StringEnum } from "@earendil-works/pi-ai";
import { Type } from "typebox";

const MAX_KNOWN_ITEMS = 5;
const MAX_TEXT_LENGTH = 120;
const DEBUG_ENABLED = !["", "0", "false", "no", "off"].includes(
  (process.env.PI_DEBUG ?? "").trim().toLowerCase(),
);

type Progress = {
  item: string;
  status: "unknown" | "active" | "complete";
};

type TaskState = {
  subgoal: string | null;
  progress: Progress[];
  known: string[];
  lastResult: string | null;
};

const progressEntry = Type.Object({
  item: Type.String({ minLength: 1, maxLength: MAX_TEXT_LENGTH }),
  status: StringEnum(["unknown", "active", "complete"] as const),
});

const stateUpdate = Type.Object({
  subgoal: Type.Optional(Type.String({ maxLength: MAX_TEXT_LENGTH })),
  progress: Type.Optional(Type.Array(progressEntry, { maxItems: MAX_KNOWN_ITEMS })),
  known: Type.Optional(
    Type.Array(Type.String({ minLength: 1, maxLength: MAX_TEXT_LENGTH }), {
      maxItems: MAX_KNOWN_ITEMS,
    }),
  ),
  last_result: Type.Optional(Type.String({ maxLength: MAX_TEXT_LENGTH })),
});

/**
 * Keep Pi's saved session intact while presenting a bounded working context to
 * the model: the newest observation prompt and its attached images only.
 */
export default function dsrfContext(pi: ExtensionAPI): void {
  let taskState = emptyTaskState();
  let robotActionRequested = false;

  pi.on("session_start", async () => {
    taskState = emptyTaskState();
    robotActionRequested = false;
    debugState("state reset", taskState);
    enableDecisionTools(pi);
  });

  // Each external Dora observation begins a fresh decision.  A successful
  // update_state call then narrows the remainder of that decision to action.
  pi.on("input", async (event) => {
    if (event.source === "rpc") {
      robotActionRequested = false;
      enableDecisionTools(pi);
    }
  });

  // A provider may emit duplicate robot_action calls in one assistant message.
  // Only the first can represent this observation; block later calls before the
  // local tool executes.  Both calls are still visible in RPC preflight events,
  // so the Python client also keeps the first action defensively.
  pi.on("tool_call", async (event) => {
    if (!isToolCallEventType("robot_action", event)) return;
    if (robotActionRequested) {
      return {
        block: true,
        reason: "Only one robot action is allowed per observation.",
        terminate: true,
      };
    }
    robotActionRequested = true;
  });

  pi.registerTool({
    name: "update_state",
    label: "Update Task State",
    description:
      "Replace the bounded task facts after a meaningful discovery or progress change. " +
      "Do not store reasoning, speculation, or an action transcript. Then call robot_action.",
    parameters: stateUpdate,
    async execute(_toolCallId, update) {
      if (Object.keys(update).length === 0) {
        throw new Error("update_state requires at least one field");
      }
      taskState = {
        subgoal: update.subgoal ?? taskState.subgoal,
        progress: update.progress ?? taskState.progress,
        known: update.known ?? taskState.known,
        lastResult: update.last_result ?? taskState.lastResult,
      };
      debugState("state updated", taskState);
      pi.setActiveTools(["robot_action"]);
      return {
        content: [{ type: "text", text: "Task state updated." }],
        details: taskState,
      };
    },
  });

  pi.on("context", async (event) => {
    const currentObservation = [...event.messages]
      .reverse()
      .find((message) => message.role === "user");
    if (!currentObservation) return { messages: [] };

    debugState("active state", taskState);
    const stateBlock = { type: "text" as const, text: formatTaskState(taskState) };
    const content =
      typeof currentObservation.content === "string"
        ? [stateBlock, { type: "text" as const, text: currentObservation.content }]
        : [stateBlock, ...currentObservation.content];
    return { messages: [{ ...currentObservation, content }] };
  });
}

function emptyTaskState(): TaskState {
  return { subgoal: null, progress: [], known: [], lastResult: null };
}

function formatTaskState(state: TaskState): string {
  const lines = ["TASK STATE"];
  if (state.subgoal) lines.push(`subgoal: ${state.subgoal}`);
  for (const entry of state.progress) {
    lines.push(`progress: ${entry.item} = ${entry.status}`);
  }
  for (const fact of state.known) lines.push(`known: ${fact}`);
  if (state.lastResult) lines.push(`last result: ${state.lastResult}`);
  if (lines.length === 1) lines.push("No task facts recorded yet.");
  return `${lines.join("\n")}\n`;
}

function debugState(label: string, state: TaskState): void {
  if (DEBUG_ENABLED) console.error(`[dsrf-context] ${label}: ${JSON.stringify(state)}`);
}

function enableDecisionTools(pi: ExtensionAPI): void {
  pi.setActiveTools(["robot_action", "update_state"]);
}
