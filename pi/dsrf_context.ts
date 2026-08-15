import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/**
 * Keep Pi's saved session intact while presenting a bounded working context to
 * the model: the newest observation prompt and its attached images only.
 */
export default function dsrfContext(pi: ExtensionAPI): void {
  pi.on("session_start", async () => {
    pi.setActiveTools(["robot_action"]);
  });

  pi.on("context", async (event) => {
    const currentObservation = [...event.messages]
      .reverse()
      .find((message) => message.role === "user");
    return { messages: currentObservation ? [currentObservation] : [] };
  });
}
