import { describe, expect } from "bun:test"
import { LATEST_PROTOCOL_VERSION } from "@modelcontextprotocol/sdk/types.js"
import { Effect } from "effect"
import fs from "node:fs/promises"
import path from "node:path"
import { cliIt } from "../lib/cli-process"

const serve = Effect.acquireRelease(
  Effect.sync(() => {
    const requests = new Map<string, Array<string | null>>()
    const http = Bun.serve({
      port: 0,
      async fetch(request) {
        const url = new URL(request.url)
        const seen = requests.get(url.pathname) ?? []
        seen.push(request.headers.get("authorization"))
        requests.set(url.pathname, seen)

        if (seen.length === 1) return new Response("Unauthorized", { status: 401 })
        if (request.method === "DELETE") return new Response(null, { status: 200 })
        if (request.method !== "POST") return new Response(null, { status: 405 })

        const body = (await request.json()) as { id?: string | number; method?: string }
        if (body.method === "notifications/initialized") return new Response(null, { status: 202 })
        return Response.json(
          {
            jsonrpc: "2.0",
            id: body.id,
            result: {
              protocolVersion: LATEST_PROTOCOL_VERSION,
              capabilities: {},
              serverInfo: { name: "static-auth-debug", version: "1.0.0" },
            },
          },
          { headers: { "Mcp-Session-Id": crypto.randomUUID() } },
        )
      },
    })
    return {
      requests,
      url: http.url,
      close: () => http.stop(true),
    }
  }),
  (server) => Effect.promise(server.close),
)

describe("opencode mcp debug", () => {
  cliIt.live(
    "never uses stored OAuth credentials when static Authorization is configured",
    ({ home, opencode }) =>
      Effect.gen(function* () {
        const server = yield* serve
        const cases = ["Authorization", "authorization", "aUtHoRiZaTiOn"].map((header, index) => {
          const name = `static-debug-${index}`
          const url = new URL(`/${name}`, server.url).toString()
          return {
            name,
            url,
            header,
            staticValue: `Bearer configured-static-${index}`,
            oauthValue: `stored-oauth-secret-${index}`,
          }
        })
        const config = {
          mcp: Object.fromEntries(
            cases.map((item) => [
              item.name,
              { type: "remote", url: item.url, headers: { [item.header]: item.staticValue } },
            ]),
          ),
        }
        const data = path.join(home, ".local", "share", "opencode")
        yield* Effect.promise(() => fs.mkdir(data, { recursive: true }))
        yield* Effect.promise(() =>
          Bun.write(
            path.join(data, "mcp-auth.json"),
            JSON.stringify(
              Object.fromEntries(
                cases.map((item) => [item.name, { tokens: { accessToken: item.oauthValue }, serverUrl: item.url }]),
              ),
            ),
          ),
        )

        for (const item of cases) {
          const result = yield* opencode.spawn(["mcp", "debug", item.name], {
            timeoutMs: 30_000,
            env: { OPENCODE_CONFIG_CONTENT: JSON.stringify(config) },
          })
          opencode.expectExit(result, 0, item.name)
          expect(result.stdout).not.toContain("Access token:")
          expect(result.stdout).not.toContain("Testing OAuth flow")
          expect(server.requests.get(new URL(item.url).pathname)).toEqual([item.staticValue])
        }
      }),
    120_000,
  )
})
