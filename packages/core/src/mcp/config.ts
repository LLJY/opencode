export * as McpConfig from "./config.js"

import { ConfigMCP } from "@opencode/schema/config/mcp"

export function oauthDisabled(config: typeof ConfigMCP.Remote.Type) {
  return (
    config.oauth === false || Object.keys(config.headers ?? {}).some((key) => key.toLowerCase() === "authorization")
  )
}
