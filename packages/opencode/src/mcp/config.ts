import { ConfigMCPV1 } from "@opencode-ai/core/v1/config/mcp"

type Remote = Extract<ConfigMCPV1.Info, { type: "remote" }>

export function oauthDisabled(config: Remote) {
  return (
    config.oauth === false || Object.keys(config.headers ?? {}).some((key) => key.toLowerCase() === "authorization")
  )
}

export * as McpConfig from "./config"
