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
const REQUIRE_STATE_UPDATE = !["", "0", "false", "no", "off"].includes(
  (process.env.PI_REQUIRE_STATE_UPDATE ?? "").trim().toLowerCase(),
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

const stateUpdate = Type.Object(
  {
    subgoal: Type.Optional(Type.String({ maxLength: MAX_TEXT_LENGTH })),
    progress: Type.Optional(
      Type.Array(progressEntry, { maxItems: MAX_KNOWN_ITEMS }),
    ),
    known: Type.Optional(
      Type.Array(Type.String({ minLength: 1, maxLength: MAX_TEXT_LENGTH }), {
        maxItems: MAX_KNOWN_ITEMS,
      }),
    ),
    last_result: Type.Optional(Type.String({ maxLength: MAX_TEXT_LENGTH })),
  },
  { minProperties: 1 },
);

/**
 * Keep Pi's saved session intact while presenting a bounded working context to
 * the model: the newest observation prompt and its attached images only.
 */
export default function dsrfContext(pi: ExtensionAPI): void {
  let taskState = emptyTaskState();
  let stateUpdateRequested = false;
  let robotActionRequested = false;

  pi.on("session_start", async () => {
    taskState = emptyTaskState();
    stateUpdateRequested = false;
    robotActionRequested = false;
    debugState("state reset", taskState);
    enableDecisionTools(pi);
  });

  // Each external Dora observation begins a fresh decision.  Stateful runs
  // require update_state to precede robot_action in the same tool-call batch.
  pi.on("input", async (event) => {
    if (event.source === "rpc") {
      stateUpdateRequested = false;
      robotActionRequested = false;
      enableDecisionTools(pi);
    }
  });

  // Tool calls are preflighted in source order.  Stateful runs therefore use
  // this gate to require exactly `update_state` followed by one robot_action.
  // Both tools terminate their result, so Pi settles after that one batch.
  pi.on("tool_call", async (event) => {
    if (event.toolName === "update_state") {
      if (stateUpdateRequested) {
        return {
          block: true,
          reason: "Only one state update is allowed per observation.",
        };
      }
      stateUpdateRequested = true;
      return;
    }
    if (!isToolCallEventType("robot_action", event)) return;
    if (REQUIRE_STATE_UPDATE && !stateUpdateRequested) {
      return {
        block: true,
        reason: "Call update_state before robot_action for this observation.",
      };
    }
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
      return {
        content: [{ type: "text", text: "Task state updated." }],
        details: taskState,
        // Stateful runs pair this with robot_action in the same batch.  Making
        // both results terminal prevents another LLM turn after the action.
        terminate: REQUIRE_STATE_UPDATE,
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
  if (REQUIRE_STATE_UPDATE) {
    pi.setActiveTools(["update_state", "robot_action"]);
    debugPhase("update_state -> robot_action");
  } else {
    pi.setActiveTools(["robot_action", "update_state"]);
    debugPhase("decision");
  }
}

function debugPhase(phase: string): void {
  if (DEBUG_ENABLED) console.error(`[dsrf-context] phase: ${phase}`);
}
