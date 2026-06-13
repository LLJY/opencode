import { describe, expect, test } from "bun:test"
import { experimentalWebSocketsEnabled } from "../../src/plugin"

describe("plugin.openai.websocket rollout", () => {
  test("forces websockets on all channels", () => {
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "local" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "dev" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "beta" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "latest" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: false, channel: "prod" })).toBe(true)
  })

  test("keeps websockets enabled when the flag is set", () => {
    expect(experimentalWebSocketsEnabled({ enabled: true, channel: "latest" })).toBe(true)
    expect(experimentalWebSocketsEnabled({ enabled: true, channel: "prod" })).toBe(true)
  })
})
