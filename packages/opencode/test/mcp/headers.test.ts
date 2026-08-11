import { describe, expect } from "bun:test"
import { Server } from "@modelcontextprotocol/sdk/server/index.js"
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js"
import { ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Cause, Effect, Exit } from "effect"
import { testEffect } from "../lib/effect"
import { McpAuth } from "../../src/mcp/auth"
import { MCP } from "../../src/mcp/index"

const it = testEffect(LayerNode.compile(LayerNode.group([MCP.node, McpAuth.node])))

const serve = Effect.acquireRelease(
  Effect.promise(async () => {
    const requests: Headers[] = []
    const protocol = new Server({ name: "headers", version: "1.0.0" }, { capabilities: { tools: {} } })
    protocol.setRequestHandler(ListToolsRequestSchema, () => Promise.resolve({ tools: [] }))
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: () => crypto.randomUUID(),
      enableJsonResponse: true,
    })
    await protocol.connect(transport)
    const http = Bun.serve({
      port: 0,
      fetch(request) {
        requests.push(new Headers(request.headers))
        return transport.handleRequest(request)
      },
    })
    return {
      requests,
      url: http.url.toString(),
      close: async () => {
        await http.stop(true)
        await protocol.close()
      },
    }
  }),
  (server) => Effect.promise(server.close),
)

describe("mcp.headers", () => {
  it.instance("configured Authorization casing overrides stored OAuth credentials", () =>
    Effect.gen(function* () {
      const mcp = yield* MCP.Service
      const auth = yield* McpAuth.Service

      for (const [index, header] of ["Authorization", "authorization", "aUtHoRiZaTiOn"].entries()) {
        const server = yield* serve
        const name = `static-auth-${index}`
        const value = `Bearer ${name}`
        yield* auth.updateTokens(name, { accessToken: "oauth-token" }, server.url)
        const result = yield* mcp.add(name, {
          type: "remote",
          url: server.url,
          headers: { [header]: value },
        })

        expect(result.status).toMatchObject({ [name]: { status: "connected" } })
        expect(server.requests.length).toBeGreaterThan(0)
        for (const headers of server.requests) {
          expect(headers.get("authorization")).toBe(value)
        }
      }
    }),
  )

  it.instance("configured Authorization rejects manual OAuth before auth setup", () =>
    Effect.gen(function* () {
      const mcp = yield* MCP.Service
      const name = "static-auth-manual"
      yield* mcp.add(name, {
        type: "remote",
        url: "not-an-mcp-url",
        enabled: false,
        headers: { authorization: "Bearer static" },
      })

      const exit = yield* mcp.startAuth(name).pipe(Effect.exit)
      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isSuccess(exit)) throw new Error("expected explicit Authorization to disable OAuth")
      const error = Cause.squash(exit.cause)
      expect(error).toBeInstanceOf(Error)
      if (error instanceof Error) expect(error.message).toContain("explicit Authorization")
    }),
  )

  it.instance("configured Authorization disables OAuth capability regardless of casing", () =>
    Effect.gen(function* () {
      const mcp = yield* MCP.Service

      for (const [index, header] of ["Authorization", "authorization", "aUtHoRiZaTiOn"].entries()) {
        const name = `static-auth-capability-${index}`
        yield* mcp.add(name, {
          type: "remote",
          url: "https://example.com/mcp",
          enabled: false,
          headers: { [header]: "Bearer static" },
        })

        expect(yield* mcp.supportsOAuth(name)).toBe(false)
      }
    }),
  )

  it.instance("headers are passed to transports when oauth is enabled (default)", () =>
    Effect.gen(function* () {
      const server = yield* serve
      const mcp = yield* MCP.Service
      const result = yield* mcp.add("test-server", {
        type: "remote",
        url: server.url,
        headers: {
          Authorization: "Bearer test-token",
          "X-Custom-Header": "custom-value",
        },
      })

      expect(result.status).toMatchObject({ "test-server": { status: "connected" } })
      expect(server.requests.length).toBeGreaterThan(0)
      for (const headers of server.requests) {
        expect(headers.get("authorization")).toBe("Bearer test-token")
        expect(headers.get("x-custom-header")).toBe("custom-value")
      }
    }),
  )

  it.instance("headers are passed to transports when oauth is explicitly disabled", () =>
    Effect.gen(function* () {
      const server = yield* serve
      const mcp = yield* MCP.Service
      const result = yield* mcp.add("test-server-no-oauth", {
        type: "remote",
        url: server.url,
        oauth: false,
        headers: {
          Authorization: "Bearer test-token",
        },
      })

      expect(result.status).toMatchObject({ "test-server-no-oauth": { status: "connected" } })
      expect(server.requests.length).toBeGreaterThan(0)
      for (const headers of server.requests) {
        expect(headers.get("authorization")).toBe("Bearer test-token")
      }
    }),
  )

  it.instance("no requestInit when headers are not provided", () =>
    Effect.gen(function* () {
      const server = yield* serve
      const mcp = yield* MCP.Service
      const result = yield* mcp.add("test-server-no-headers", {
        type: "remote",
        url: server.url,
      })

      expect(result.status).toMatchObject({ "test-server-no-headers": { status: "connected" } })
      expect(server.requests.length).toBeGreaterThan(0)
      for (const headers of server.requests) {
        expect(headers.has("authorization")).toBe(false)
        expect(headers.has("x-custom-header")).toBe(false)
      }
    }),
  )
})
