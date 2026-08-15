import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { StringEnum } from "@earendil-works/pi-ai";
import { Type } from "typebox";

const motion = Type.String({ minLength: 1 });
const direction = StringEnum(["forward", "backward", "left", "right"] as const);
const waypoint = Type.Tuple([
  Type.Integer({ minimum: 0, maximum: 1000 }),
  Type.Integer({ minimum: 0, maximum: 1000 }),
]);

export default function (pi: ExtensionAPI) {
  const directionMode = process.env.DSRF_COMMAND_MODE === "direction";
  const parameters = directionMode
    ? Type.Union([
        Type.Object({
          motion: Type.Literal("stand"),
          direction,
        }),
        Type.Object({ motion: Type.Literal("walk"), direction }),
      ])
    : Type.Object({
        motion,
        waypoints_2d: Type.Array(waypoint),
      });

  pi.registerTool({
    name: "robot_action",
    label: "Robot Action",
    description:
      "Choose exactly one next action for the robot from the current observation.",
    parameters,
    async execute(_toolCallId, params) {
      return {
        content: [{ type: "text", text: "Robot action accepted." }],
        details: { action: params },
        terminate: true,
      };
    },
  });
}
