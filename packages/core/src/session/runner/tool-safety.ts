import { SessionError } from "@opencode/schema/session-error"

// A terminal error is not proof that an external effect failed. Keep this durable
// marker distinct from settled Tool.Error results and never infer safe replay from it.
export const TOOL_OUTCOME_UNKNOWN: SessionError.Error = {
  type: "tool.outcome-unknown",
  message: "Tool outcome is unknown. Reconcile the affected operation before continuing.",
}
