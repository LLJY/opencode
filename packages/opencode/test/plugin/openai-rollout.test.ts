import { describe, expect, test } from "bun:test"
import { experimentalWebSocketsEnabled } from "../../src/plugin"

describe("plugin.openai.websocket rollout", () => {
  test("enables websockets by default on every channel", () => {
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "local" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "dev" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "beta" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "latest" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "prod" })).toBe(true)
  })

  test("keeps the override enabled even when the flag is set", () => {
    expect(experimentalWebSocketsEnabled({ enabled: true, channel: "latest" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: true, channel: "prod" })).toBe(true)
  })
})
