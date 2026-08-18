import { describe, expect, test } from "bun:test"
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js"
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js"
import { Server } from "@modelcontextprotocol/sdk/server/index.js"
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js"
import {
  CallToolRequestSchema,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  type JSONRPCMessage,
} from "@modelcontextprotocol/sdk/types.js"
import type { FetchLike } from "@modelcontextprotocol/sdk/shared/transport.js"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Cause, Effect, Exit } from "effect"
import { MCP } from "../../src/mcp/index"
import { McpRateLimit } from "../../src/mcp/rate-limit"
import { pollWithTimeout, testEffect } from "../lib/effect"

const it = testEffect(LayerNode.compile(MCP.node))
const now = Date.UTC(2026, 0, 2, 3, 4, 5)

describe("McpRateLimit.delay", () => {
  test("parses Retry-After delta-seconds and HTTP-date before reset headers", () => {
    expect(McpRateLimit.delay(new Headers({ "retry-after": "2", "ratelimit-reset": "9" }), 0, { now, random: 0 })).toBe(
      2_000,
    )
    expect(
      McpRateLimit.delay(new Headers({ "retry-after": new Date(now + 4_000).toUTCString() }), 0, {
        now,
        random: 0,
      }),
    ).toBe(4_000)
  })

  test("uses applicable reset headers when Retry-After is absent or invalid", () => {
    expect(McpRateLimit.delay(new Headers({ "ratelimit-reset": "3" }), 0, { now, random: 0 })).toBe(3_000)
    expect(McpRateLimit.delay(new Headers({ "x-ratelimit-reset-after": "1.25" }), 0, { now, random: 0 })).toBe(1_250)
    expect(
      McpRateLimit.delay(new Headers({ "retry-after": "invalid", "x-ratelimit-reset": String(now / 1_000 + 5) }), 0, {
        now,
        random: 0,
      }),
    ).toBe(5_000)
  })

  test("uses capped exponential backoff with deterministic jitter", () => {
    expect(McpRateLimit.delay(new Headers(), 0, { now, random: 0 })).toBe(500)
    expect(McpRateLimit.delay(new Headers({ "retry-after": "0" }), 1, { now, random: 0 })).toBe(1_000)
    expect(McpRateLimit.delay(new Headers({ "ratelimit-reset": "0" }), 2, { now, random: 0 })).toBe(2_000)
    expect(McpRateLimit.delay(new Headers({ "x-ratelimit-reset-after": "0" }), 3, { now, random: 0 })).toBe(4_000)
    expect(McpRateLimit.delay(new Headers({ "x-ratelimit-reset": String(now / 1_000) }), 4, { now, random: 0 })).toBe(
      8_000,
    )
    expect(McpRateLimit.delay(new Headers(), 0, { now, random: 1 })).toBe(1_000)
    expect(McpRateLimit.delay(new Headers(), 10, { now, random: 0 })).toBe(15_000)
    expect(McpRateLimit.delay(new Headers(), 10, { now, random: 1 })).toBe(30_000)
  })

  test.each([".2050", "+.2050", "-.2050", "1.5", "2.0"])(
    "rejects numeric-looking malformed Retry-After value %s",
    (value) => {
      expect(McpRateLimit.delay(new Headers({ "retry-after": value }), 0, { now, random: 0 })).toBe(500)
    },
  )

  // A parseable header is authoritative even when it asks for no wait, so it must not
  // hand control to a header the server ranked lower.
  test("a Retry-After of zero preempts reset headers", () => {
    expect(
      McpRateLimit.delay(new Headers({ "retry-after": "0", "ratelimit-reset": "9" }), 0, { now, random: 0 }),
    ).toBe(500)
    expect(
      McpRateLimit.delay(new Headers({ "retry-after": "0", "x-ratelimit-reset-after": "9" }), 1, { now, random: 0 }),
    ).toBe(1_000)
  })

  test("an elapsed Retry-After HTTP-date preempts reset headers", () => {
    expect(
      McpRateLimit.delay(
        new Headers({ "retry-after": new Date(now - 4_000).toUTCString(), "ratelimit-reset": "9" }),
        0,
        { now, random: 0 },
      ),
    ).toBe(500)
  })

  test("a zero reset header preempts the lower-priority reset headers behind it", () => {
    expect(
      McpRateLimit.delay(new Headers({ "ratelimit-reset": "0", "x-ratelimit-reset-after": "9" }), 0, {
        now,
        random: 0,
      }),
    ).toBe(500)
    expect(
      McpRateLimit.delay(
        new Headers({ "x-ratelimit-reset-after": "0", "x-ratelimit-reset": String(now / 1_000 + 9) }),
        1,
        { now, random: 0 },
      ),
    ).toBe(1_000)
  })

  test("caps oversized header hints at the runtime timer limit", () => {
    expect(McpRateLimit.delay(new Headers({ "retry-after": "999999999" }), 0, { now, random: 0 })).toBe(2_147_483_647)
    expect(McpRateLimit.delay(new Headers({ "ratelimit-reset": "999999999" }), 0, { now, random: 0 })).toBe(
      2_147_483_647,
    )
  })

  test.each(["January 1, 2099", "Jan 1, 2099", "2099-01-01T00:00:00Z", "Tomorrow", "sun, 06 nov 1994 08:49:37 gmt"])(
    "rejects alphabetic non-HTTP-date Retry-After value %p",
    (value) => {
      expect(McpRateLimit.delay(new Headers({ "retry-after": value }), 0, { now, random: 0 })).toBe(500)
      expect(
        McpRateLimit.delay(new Headers({ "retry-after": value, "ratelimit-reset": "9" }), 0, { now, random: 0 }),
      ).toBe(9_000)
    },
  )

  // Well-formed grammar, invalid instant: these must not be trusted as deadlines.
  test.each([
    // 2026-01-02 is a Friday, not a Monday.
    "Mon, 02 Jan 2026 03:04:09 GMT",
    "Fri, 30 Feb 2024 12:00:00 GMT",
    "Fri, 02 Jan 0026 03:04:09 GMT",
  ])("falls back when Retry-After is a well-formed but invalid HTTP-date %p", (value) => {
    expect(McpRateLimit.delay(new Headers({ "retry-after": value }), 0, { now, random: 0 })).toBe(500)
    expect(
      McpRateLimit.delay(new Headers({ "retry-after": value, "ratelimit-reset": "9" }), 0, { now, random: 0 }),
    ).toBe(9_000)
  })

  // `now` is 2026-01-02T03:04:05Z, so each of these is four seconds ahead.
  test.each([
    ["rfc850", "Friday, 02-Jan-26 03:04:09 GMT"],
    ["asctime padded day", "Fri Jan 02 03:04:09 2026"],
    ["asctime single digit day", "Fri Jan  2 03:04:09 2026"],
  ])("accepts the obsolete %s HTTP-date form", (_form, value) => {
    expect(McpRateLimit.delay(new Headers({ "retry-after": value }), 0, { now, random: 0 })).toBe(4_000)
  })
})

describe("McpRateLimit.RetryTransport", () => {
  test("repeats definite MCP 429 responses until success", async () => {
    const statuses = [429, 429, 200]
    const waits: number[] = []
    let requests = 0
    let cancelled = 0
    const transport = fakeTransport(
      () => {
        const status = statuses[requests++] ?? 200
        return Promise.resolve(
          new Response(
            status === 429
              ? new ReadableStream({
                  cancel() {
                    cancelled++
                  },
                })
              : null,
            { status, headers: { "x-ratelimit-reset-after": "0.001" } },
          ),
        )
      },
      (delay) => {
        waits.push(delay)
        return Promise.resolve()
      },
    )

    await transport.send({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} })

    expect(requests).toBe(3)
    expect(waits).toEqual([1, 1])
    expect(cancelled).toBe(2)
  })

  test("does not retry non-429 responses", async () => {
    let requests = 0
    const transport = fakeTransport(() => {
      requests++
      return Promise.resolve(new Response("unavailable", { status: 503 }))
    })

    await transport.send({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} })

    expect(requests).toBe(1)
  })

  test("transport close interrupts retry when 429 body cancellation stalls", async () => {
    const entered = Promise.withResolvers<void>()
    const transport = fakeTransport(
      () =>
        Promise.resolve(
          new Response(
            new ReadableStream({
              cancel() {
                return new Promise(() => {})
              },
            }),
            { status: 429, headers: { "retry-after": "30" } },
          ),
        ),
      (_delay, signals) => {
        entered.resolve()
        return new Promise((resolve, reject) => {
          const signal = AbortSignal.any([...signals])
          const abort = () => reject(signal.reason)
          signal.addEventListener("abort", abort, { once: true })
          if (signal.aborted) abort()
        })
      },
    )
    const send = transport.send({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} })
    await entered.promise

    await transport.close()

    await expect(send).rejects.toBeDefined()
  })

  test("aborts a hung in-flight fetch when the transport closes", async () => {
    const entered = Promise.withResolvers<void>()
    let requests = 0
    const transport = fakeTransport((_url, init) => {
      requests++
      entered.resolve()
      return hangUntilAborted(init?.signal)
    })
    const send = transport.send({ jsonrpc: "2.0", id: 1, method: "tools/call", params: {} })
    await entered.promise

    await transport.close()

    expect(await settled(send)).toMatchObject({ status: "rejected", error: { name: "AbortError" } })
    expect(requests).toBe(1)
  })

  test("aborts a hung in-flight fetch when the caller signal aborts", async () => {
    const entered = Promise.withResolvers<void>()
    const controller = new AbortController()
    const endpoint = new URL("https://mcp.example.test")
    let requests = 0
    const transport = new McpRateLimit.RetryTransport(
      endpoint,
      (retry) => ({
        start: () => Promise.resolve(),
        close: () => Promise.resolve(),
        finishAuth: () => Promise.resolve(),
        send: (message) =>
          retry(endpoint, {
            method: "POST",
            body: JSON.stringify(message),
            signal: controller.signal,
          }).then(() => {}),
      }),
      {
        fetch: (_url, init) => {
          requests++
          entered.resolve()
          return hangUntilAborted(init?.signal)
        },
      },
    )
    const send = transport.send({ jsonrpc: "2.0", id: 1, method: "tools/call", params: {} })
    await entered.promise

    controller.abort("cancelled by caller")

    try {
      expect(await settled(send)).toEqual({ status: "rejected", error: "cancelled by caller" })
      expect(requests).toBe(1)
    } finally {
      await transport.close()
    }
  })

  test("aborts a hung in-flight fetch when the SDK marks the request inactive", async () => {
    const entered = Promise.withResolvers<void>()
    let active = true
    let requests = 0
    const transport = fakeTransport((_url, init) => {
      requests++
      entered.resolve()
      return hangUntilAborted(init?.signal)
    })
    const send = transport.send(
      { jsonrpc: "2.0", id: 1, method: "tools/call", params: {} },
      { isRequestActive: () => active },
    )
    await entered.promise

    active = false

    try {
      expect(await settled(send)).toMatchObject({
        status: "rejected",
        error: { name: "AbortError", message: "MCP request was cancelled" },
      })
      expect(requests).toBe(1)
    } finally {
      await transport.close()
    }
  })

  // Headers are not the end of the request: the SDK reads the JSON reply afterwards,
  // so a server that answers 200 and then stalls must still observe cancellation.
  test("cancels a stalled successful JSON body when the SDK marks the request inactive", async () => {
    const headers = Promise.withResolvers<void>()
    let active = true
    let cancelled = 0
    let requests = 0
    const transport = bodyTransport(
      () => {
        requests++
        return Promise.resolve(
          new Response(
            new ReadableStream<Uint8Array>({
              pull: () => new Promise<void>(() => {}),
              cancel() {
                cancelled++
              },
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        )
      },
      async (response) => {
        headers.resolve()
        await response.text()
      },
    )
    const send = transport.send(
      { jsonrpc: "2.0", id: 1, method: "tools/call", params: {} },
      { isRequestActive: () => active },
    )
    await headers.promise

    active = false

    try {
      expect(await settled(send)).toMatchObject({
        status: "rejected",
        error: { name: "AbortError", message: "MCP request was cancelled" },
      })
      expect(cancelled).toBe(1)
      expect(requests).toBe(1)
    } finally {
      await transport.close()
    }
  })

  test("cancels a stalled POST SSE body when the SDK marks the request inactive", async () => {
    const opened = Promise.withResolvers<void>()
    const encoder = new TextEncoder()
    let active = true
    let cancelled = 0
    let requests = 0
    const transport = bodyTransport(
      () => {
        requests++
        return Promise.resolve(
          new Response(
            new ReadableStream<Uint8Array>({
              start(controller) {
                controller.enqueue(encoder.encode(": open\n\n"))
              },
              pull: () => new Promise<void>(() => {}),
              cancel() {
                cancelled++
              },
            }),
            { status: 200, headers: { "content-type": "text/event-stream" } },
          ),
        )
      },
      async (response) => {
        const body = response.body
        if (!body) throw new Error("expected an SSE body")
        const reader = body.getReader()
        await reader.read()
        opened.resolve()
        await reader.read()
      },
    )
    const send = transport.send(
      { jsonrpc: "2.0", id: 1, method: "tools/call", params: {} },
      { isRequestActive: () => active },
    )
    await opened.promise

    active = false

    try {
      expect(await settled(send)).toMatchObject({
        status: "rejected",
        error: { name: "AbortError", message: "MCP request was cancelled" },
      })
      expect(cancelled).toBe(1)
      expect(requests).toBe(1)
    } finally {
      await transport.close()
    }
  })

  // Watching the body for request-active cancellation must not unhook the signals the
  // fetch was already composed from.
  test("keeps transport close wired to a watched successful body", async () => {
    const headers = Promise.withResolvers<void>()
    let requests = 0
    const transport = bodyTransport(
      (_url, init) => {
        requests++
        return Promise.resolve(
          new Response(
            new ReadableStream<Uint8Array>({
              start(controller) {
                init?.signal?.addEventListener(
                  "abort",
                  () => controller.error(new DOMException("This operation was aborted", "AbortError")),
                  { once: true },
                )
              },
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        )
      },
      async (response) => {
        headers.resolve()
        await response.text()
      },
    )
    const send = transport.send(
      { jsonrpc: "2.0", id: 1, method: "tools/call", params: {} },
      { isRequestActive: () => true },
    )
    await headers.promise

    await transport.close()

    expect(await settled(send)).toMatchObject({ status: "rejected", error: { name: "AbortError" } })
    expect(requests).toBe(1)
  })

  // The interval outlives a promise that was never created, so clearing it off the
  // fetch result leaks a timer that keeps polling a request that already failed.
  test("clears the request-active poll when a custom fetch throws synchronously", async () => {
    const failure = new Error("fetch exploded before returning a promise")
    let polls = 0
    const transport = fakeTransport(() => {
      throw failure
    })

    await expect(
      transport.send(
        { jsonrpc: "2.0", id: 1, method: "tools/call", params: {} },
        {
          isRequestActive: () => {
            polls++
            return true
          },
        },
      ),
    ).rejects.toBe(failure)

    const observed = polls
    // Four poll intervals: a leaked timer would have run several more times by now.
    await Bun.sleep(100)
    expect(polls).toBe(observed)
  })

  test("inner transport close interrupts pending retry backoff", async () => {
    const entered = Promise.withResolvers<void>()
    const closed = Promise.withResolvers<() => void>()
    const endpoint = new URL("https://mcp.example.test")
    const transport = new McpRateLimit.RetryTransport(
      endpoint,
      (retry) => ({
        start: () => Promise.resolve(),
        close: () => Promise.resolve(),
        finishAuth: () => Promise.resolve(),
        send: (message) => retry(endpoint, { method: "POST", body: JSON.stringify(message) }).then(() => {}),
        set onclose(value: (() => void) | undefined) {
          if (value) closed.resolve(value)
        },
      }),
      {
        fetch: () => Promise.resolve(new Response(null, { status: 429, headers: { "retry-after": "30" } })),
        wait: (_delay, signals) => {
          entered.resolve()
          return new Promise((resolve, reject) => {
            const signal = AbortSignal.any([...signals])
            const abort = () => reject(signal.reason)
            signal.addEventListener("abort", abort, { once: true })
            if (signal.aborted) abort()
          })
        },
      },
    )
    const send = transport.send({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} })
    await entered.promise

    const close = await closed.promise
    close()

    const result = await Promise.race([
      send.then(
        () => ({ status: "resolved" as const }),
        (error) => ({ status: "rejected" as const, error }),
      ),
      Bun.sleep(25).then(() => ({ status: "pending" as const })),
    ])
    await transport.close()
    expect(result).toMatchObject({ status: "rejected", error: { name: "AbortError" } })
  })

  test("forwards optional Streamable HTTP session APIs", async () => {
    const resumed: string[] = []
    let terminated = 0
    let protocolVersion = "2025-03-26"
    const endpoint = new URL("https://mcp.example.test")
    const transport = new McpRateLimit.RetryTransport(endpoint, () => ({
      start: () => Promise.resolve(),
      close: () => Promise.resolve(),
      finishAuth: () => Promise.resolve(),
      send: () => Promise.resolve(),
      terminateSession: () => {
        terminated++
        return Promise.resolve()
      },
      resumeStream: (lastEventId) => {
        resumed.push(lastEventId)
        return Promise.resolve()
      },
      setProtocolVersion: (version) => {
        protocolVersion = version
      },
      get protocolVersion() {
        return protocolVersion
      },
    }))

    transport.setProtocolVersion("2025-06-18")
    await transport.resumeStream("event-42")
    await transport.terminateSession()

    expect(transport.protocolVersion).toBe("2025-06-18")
    expect(resumed).toEqual(["event-42"])
    expect(terminated).toBe(1)
  })

  test("bounds always-429 Streamable HTTP SSE resumption GET", async () => {
    const endpoint = new URL("https://mcp.example.test")
    const methods: string[] = []
    const eventIds: Array<string | null> = []
    const cancelled: number[] = []
    const errors: Error[] = []
    let attempts = 0
    const transport = new McpRateLimit.RetryTransport(
      endpoint,
      (fetch) => new StreamableHTTPClientTransport(endpoint, { fetch }),
      {
        fetch: (_url, init) => {
          const attempt = ++attempts
          methods.push(init?.method ?? "GET")
          eventIds.push(new Headers(init?.headers).get("last-event-id"))
          return Promise.resolve(
            new Response(
              new ReadableStream({
                cancel() {
                  cancelled.push(attempt)
                },
              }),
              {
                status: 429,
                statusText: "Too Many Requests",
                headers: { "x-ratelimit-reset-after": "0.001" },
              },
            ),
          )
        },
      },
    )
    transport.onerror = (error) => errors.push(error)
    await transport.start()

    try {
      await expect(transport.resumeStream("event-42")).rejects.toMatchObject({ code: 429 })
      expect(attempts).toBe(4)
      expect(methods).toEqual(["GET", "GET", "GET", "GET"])
      expect(eventIds).toEqual(["event-42", "event-42", "event-42", "event-42"])
      expect(cancelled).toEqual([1, 2, 3, 4])
      expect(errors).toHaveLength(1)
      expect(errors[0]).toMatchObject({ code: 429 })
    } finally {
      await transport.close()
    }
  })

  test("bounds always-429 background Streamable HTTP SSE GET", async () => {
    const endpoint = new URL("https://mcp.example.test")
    const failed = Promise.withResolvers<Error>()
    const cancelled: number[] = []
    let gets = 0
    let posts = 0
    const transport = new McpRateLimit.RetryTransport(
      endpoint,
      (fetch) => new StreamableHTTPClientTransport(endpoint, { fetch }),
      {
        fetch: (_url, init) => {
          if (init?.method === "POST") {
            posts++
            return Promise.resolve(new Response(null, { status: 202 }))
          }

          const attempt = ++gets
          return Promise.resolve(
            new Response(
              new ReadableStream({
                cancel() {
                  cancelled.push(attempt)
                },
              }),
              {
                status: 429,
                statusText: "Too Many Requests",
                headers: { "x-ratelimit-reset-after": "0.001" },
              },
            ),
          )
        },
      },
    )
    transport.onerror = failed.resolve
    await transport.start()

    try {
      await transport.send({ jsonrpc: "2.0", method: "notifications/initialized" })
      await expect(failed.promise).resolves.toMatchObject({ code: 429 })
      expect(posts).toBe(1)
      expect(gets).toBe(4)
      expect(cancelled).toEqual([1, 2, 3, 4])
    } finally {
      await transport.close()
    }
  })

  test("retries cancellation notifications and session termination after explicit 429", async () => {
    const attempts = new Map<string, number>()
    const endpoint = new URL("https://mcp.example.test")
    const transport = new McpRateLimit.RetryTransport(
      endpoint,
      (retry) => ({
        start: () => Promise.resolve(),
        close: () => Promise.resolve(),
        finishAuth: () => Promise.resolve(),
        send: (message) => retry(endpoint, { method: "POST", body: JSON.stringify(message) }).then(() => {}),
        terminateSession: () => retry(endpoint, { method: "DELETE" }).then(() => {}),
      }),
      {
        fetch: (_url, init) => {
          const method = init?.method === "DELETE" ? "delete" : "notification"
          const count = (attempts.get(method) ?? 0) + 1
          attempts.set(method, count)
          return Promise.resolve(new Response(null, { status: count === 1 ? 429 : 200 }))
        },
        wait: () => Promise.resolve(),
      },
    )

    await transport.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: 1 } })
    await transport.terminateSession()

    expect(attempts).toEqual(
      new Map([
        ["notification", 2],
        ["delete", 2],
      ]),
    )
  })

  test("returns the fourth notification 429 for the SDK to report", async () => {
    const endpoint = new URL("https://mcp.example.test")
    const overrun = new Error("notification exceeded its retry budget")
    let attempts = 0
    let cancelled = 0
    let waits = 0
    const transport = new McpRateLimit.RetryTransport(
      endpoint,
      (fetch) => new StreamableHTTPClientTransport(endpoint, { fetch }),
      {
        fetch: () => {
          attempts++
          return Promise.resolve(
            new Response(
              new ReadableStream({
                pull(controller) {
                  controller.enqueue(new TextEncoder().encode("rate limited"))
                  controller.close()
                },
                cancel() {
                  cancelled++
                },
              }),
              { status: 429 },
            ),
          )
        },
        wait: () => {
          waits++
          return waits === 4 ? Promise.reject(overrun) : Promise.resolve()
        },
      },
    )
    await transport.start()

    try {
      await expect(
        transport.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: 1 } }),
      ).rejects.toMatchObject({ code: 429 })
      expect(attempts).toBe(4)
      expect(waits).toBe(3)
      expect(cancelled).toBe(3)
    } finally {
      await transport.close()
    }
  })

  test("does not replay an ambiguous fetch failure", async () => {
    const failure = new Error("connection reset after send")
    let requests = 0
    const transport = fakeTransport(() => {
      requests++
      return Promise.reject(failure)
    })

    await expect(transport.send({ jsonrpc: "2.0", id: 1, method: "tools/call", params: {} })).rejects.toBe(failure)
    expect(requests).toBe(1)
  })

  test("does not classify OAuth discovery GETs as MCP SSE requests", async () => {
    let requests = 0
    let waits = 0
    const endpoint = new URL("https://mcp.example.test/mcp")
    const transport = new McpRateLimit.RetryTransport(
      endpoint,
      (retry) => ({
        start: async () => {
          await retry("https://auth.example.test/.well-known/oauth-authorization-server", {
            headers: { accept: "text/event-stream" },
          })
        },
        close: () => Promise.resolve(),
        finishAuth: () => Promise.resolve(),
        send: () => Promise.resolve(),
      }),
      {
        fetch: () => {
          requests++
          return Promise.resolve(new Response("rate limited", { status: 429 }))
        },
        wait: () => {
          waits++
          return Promise.resolve()
        },
      },
    )

    await transport.start()
    expect(requests).toBe(1)
    expect(waits).toBe(0)
  })

  test("keeps concurrent request cancellation state isolated", async () => {
    const firstWait = Promise.withResolvers<void>()
    const secondWait = Promise.withResolvers<void>()
    const firstEntered = Promise.withResolvers<void>()
    const secondEntered = Promise.withResolvers<void>()
    const attempts = new Map<number, number>()
    let firstActive = true
    const isFirstActive = () => firstActive
    const isSecondActive = () => true
    const transport = fakeTransport(
      (_url, init) => {
        const id = messageId(init?.body)
        const attempt = (attempts.get(id) ?? 0) + 1
        attempts.set(id, attempt)
        return Promise.resolve(new Response("", { status: attempt === 1 ? 429 : 200 }))
      },
      (_delay, _signals, active) => {
        if (active === isFirstActive) {
          firstEntered.resolve()
          return firstWait.promise
        }
        secondEntered.resolve()
        return secondWait.promise
      },
    )
    const first = transport.send(
      { jsonrpc: "2.0", id: 1, method: "tools/call", params: {} },
      { isRequestActive: isFirstActive },
    )
    const second = transport.send(
      { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} },
      { isRequestActive: isSecondActive },
    )
    await Promise.all([firstEntered.promise, secondEntered.promise])

    firstActive = false
    firstWait.resolve()
    secondWait.resolve()

    await expect(first).rejects.toMatchObject({ name: "AbortError" })
    await expect(second).resolves.toBeUndefined()
    expect(attempts).toEqual(
      new Map([
        [1, 1],
        [2, 2],
      ]),
    )
  })

  test("retries legacy SSE connection and message POST responses", async () => {
    const encoder = new TextEncoder()
    let gets = 0
    let posts = 0
    const http = Bun.serve({
      port: 0,
      fetch(request) {
        if (request.method === "POST") {
          posts++
          if (posts === 1)
            return new Response("rate limited", { status: 429, headers: { "x-ratelimit-reset-after": "0.001" } })
          return new Response(null, { status: 202 })
        }

        gets++
        if (gets === 1)
          return new Response("rate limited", { status: 429, headers: { "x-ratelimit-reset-after": "0.001" } })
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(encoder.encode("event: endpoint\ndata: /messages\n\n"))
            },
          }),
          { headers: { "content-type": "text/event-stream" } },
        )
      },
    })
    const endpoint = new URL("/sse", http.url)
    const transport = new McpRateLimit.RetryTransport(endpoint, (fetch) => new SSEClientTransport(endpoint, { fetch }))

    try {
      await transport.start()
      await transport.send({ jsonrpc: "2.0", method: "notifications/initialized" })
      expect(gets).toBe(2)
      expect(posts).toBe(2)
    } finally {
      await transport.close()
      http.stop(true)
    }
  })
})

it.instance("recovers initialization, catalogs, prompts, resources, and tools after HTTP 429", () =>
  Effect.gen(function* () {
    const server = yield* rateLimitedServer
    server.limit("initialize", 2)
    server.limit("tools/list", 1)
    server.limit("prompts/list", 1)
    server.limit("resources/list", 1)
    server.limit("resources/templates/list", 1)
    server.limit("prompts/get", 1)
    server.limit("resources/read", 1)
    server.limit("tools/call", 1)
    const mcp = yield* MCP.Service

    const added = yield* mcp.add("limited", remote(server.url))
    expect(added.status).toMatchObject({ limited: { status: "connected" } })
    expect(Object.keys(yield* mcp.prompts())).toEqual(["limited:test-prompt"])
    expect(Object.keys(yield* mcp.resources())).toEqual(["limited:test://resource"])
    expect(Object.keys(yield* mcp.resourceTemplates())).toEqual(["limited:test://{id}"])
    expect(yield* mcp.getPrompt("limited", "test-prompt")).toBeDefined()
    expect(yield* mcp.readResource("limited", "test://resource")).toBeDefined()

    const tool = (yield* mcp.tools()).limited_rate_tool
    if (!tool) throw new Error("rate-limited tool was not listed")
    const result = yield* Effect.promise(() => tool.client.callTool({ name: tool.def.name, arguments: {} }))
    expect(result.content).toEqual([{ type: "text", text: "tool result" }])

    expect(server.count("initialize")).toBe(3)
    for (const method of [
      "tools/list",
      "prompts/list",
      "resources/list",
      "resources/templates/list",
      "prompts/get",
      "resources/read",
      "tools/call",
    ]) {
      expect(server.count(method)).toBe(2)
    }
  }),
)

it.instance("stops a 429 wait when a tool request is aborted", () =>
  Effect.gen(function* () {
    const server = yield* rateLimitedServer
    const mcp = yield* MCP.Service
    yield* mcp.add("abort", remote(server.url))
    const tool = (yield* mcp.tools()).abort_rate_tool
    if (!tool) throw new Error("rate-limited tool was not listed")
    server.limit("tools/call", Number.POSITIVE_INFINITY, "30")
    const controller = new AbortController()
    const call = tool.client.callTool({ name: tool.def.name, arguments: {} }, undefined, {
      signal: controller.signal,
      timeout: 5_000,
    })

    yield* pollWithTimeout(
      Effect.sync(() => (server.count("tools/call") === 1 ? true : undefined)),
      "tool request did not enter rate-limit wait",
    )
    controller.abort("cancelled by test")
    const exit = yield* Effect.tryPromise({
      try: () => call,
      catch: (error) => error,
    }).pipe(Effect.exit)

    expect(Exit.isFailure(exit)).toBe(true)
    expect(server.count("tools/call")).toBe(1)
  }),
)

it.instance("keeps 429 backoff inside the configured tool request timeout", () =>
  Effect.gen(function* () {
    const server = yield* rateLimitedServer
    const mcp = yield* MCP.Service
    yield* mcp.add("timeout", remote(server.url))
    const tool = (yield* mcp.tools()).timeout_rate_tool
    if (!tool) throw new Error("rate-limited tool was not listed")
    server.limit("tools/call", Number.POSITIVE_INFINITY, "30")

    const exit = yield* Effect.tryPromise({
      try: () => tool.client.callTool({ name: tool.def.name, arguments: {} }, undefined, { timeout: 40 }),
      catch: (error) => error,
    }).pipe(Effect.exit)

    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) {
      const error = Cause.squash(exit.cause)
      expect(error).toBeInstanceOf(Error)
      if (error instanceof Error) expect(error.message).toContain("Request timed out")
    }
    expect(server.count("tools/call")).toBe(1)
  }),
)

function fakeTransport(fetch: FetchLike, wait?: McpRateLimit.Wait) {
  const endpoint = new URL("https://mcp.example.test")
  return new McpRateLimit.RetryTransport(
    endpoint,
    (retry) => ({
      start: () => Promise.resolve(),
      close: () => Promise.resolve(),
      finishAuth: () => Promise.resolve(),
      send: async (message: JSONRPCMessage) => {
        await retry(endpoint, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(message),
        })
      },
    }),
    { fetch, wait },
  )
}

// Hands the returned response to the test so it can consume the body the way the SDK
// does, instead of discarding it the moment the headers arrive.
function bodyTransport(fetch: FetchLike, consume: (response: Response) => Promise<void>) {
  const endpoint = new URL("https://mcp.example.test")
  return new McpRateLimit.RetryTransport(
    endpoint,
    (retry) => ({
      start: () => Promise.resolve(),
      close: () => Promise.resolve(),
      finishAuth: () => Promise.resolve(),
      send: async (message: JSONRPCMessage) => {
        const response = await retry(endpoint, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(message),
        })
        await consume(response)
      },
    }),
    { fetch },
  )
}

// Models a request the server never answers: it settles only when the fetch signal
// aborts, so a retry loop that cannot abort its own fetch simply never finishes.
function hangUntilAborted(signal: AbortSignal | null | undefined) {
  return new Promise<Response>((_resolve, reject) => {
    if (!signal) return
    const abort = () => reject(signal.reason)
    signal.addEventListener("abort", abort, { once: true })
    if (signal.aborted) abort()
  })
}

// A send that can never be aborted would otherwise stall the suite instead of
// failing, so the assertion is taken against a bounded outcome.
function settled(promise: Promise<unknown>) {
  return Promise.race([
    promise.then(
      () => ({ status: "resolved" as const }),
      (error: unknown) => ({ status: "rejected" as const, error }),
    ),
    Bun.sleep(1_000).then(() => ({ status: "pending" as const })),
  ])
}

function messageId(body: BodyInit | null | undefined) {
  if (typeof body !== "string") throw new Error("expected JSON-RPC request body")
  const value: unknown = JSON.parse(body)
  if (typeof value !== "object" || value === null || !("id" in value) || typeof value.id !== "number") {
    throw new Error("expected numeric JSON-RPC request ID")
  }
  return value.id
}

const rateLimitedServer = Effect.acquireRelease(
  Effect.promise(async () => {
    const protocol = new Server(
      { name: "rate-limit-test", version: "1.0.0" },
      { capabilities: { tools: {}, prompts: {}, resources: {} } },
    )
    protocol.setRequestHandler(ListToolsRequestSchema, () =>
      Promise.resolve({ tools: [{ name: "rate_tool", inputSchema: { type: "object" } }] }),
    )
    protocol.setRequestHandler(ListPromptsRequestSchema, () => Promise.resolve({ prompts: [{ name: "test-prompt" }] }))
    protocol.setRequestHandler(ListResourcesRequestSchema, () =>
      Promise.resolve({ resources: [{ name: "test-resource", uri: "test://resource" }] }),
    )
    protocol.setRequestHandler(ListResourceTemplatesRequestSchema, () =>
      Promise.resolve({ resourceTemplates: [{ name: "test-template", uriTemplate: "test://{id}" }] }),
    )
    protocol.setRequestHandler(GetPromptRequestSchema, () =>
      Promise.resolve({ messages: [{ role: "user", content: { type: "text", text: "prompt result" } }] }),
    )
    protocol.setRequestHandler(ReadResourceRequestSchema, ({ params }) =>
      Promise.resolve({ contents: [{ uri: params.uri, text: "resource result" }] }),
    )
    protocol.setRequestHandler(CallToolRequestSchema, () =>
      Promise.resolve({ content: [{ type: "text" as const, text: "tool result" }] }),
    )
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: () => crypto.randomUUID(),
      enableJsonResponse: true,
    })
    await protocol.connect(transport)
    const limits = new Map<string, { remaining: number; retryAfter: string }>()
    const counts = new Map<string, number>()
    const http = Bun.serve({
      port: 0,
      async fetch(request) {
        const method = await requestMethod(request)
        counts.set(method, (counts.get(method) ?? 0) + 1)
        const limit = limits.get(method)
        if (limit && limit.remaining > 0) {
          limit.remaining--
          return new Response("rate limited", {
            status: 429,
            headers: { "x-ratelimit-reset-after": limit.retryAfter },
          })
        }
        return transport.handleRequest(request)
      },
    })

    return {
      url: http.url.toString(),
      limit(method: string, remaining: number, retryAfter = "0.001") {
        limits.set(method, { remaining, retryAfter })
      },
      count(method: string) {
        return counts.get(method) ?? 0
      },
      close: async () => {
        await protocol.close().catch(() => {})
        http.stop(true)
      },
    }
  }),
  (server) => Effect.promise(server.close),
)

async function requestMethod(request: Request) {
  if (request.method !== "POST") return request.method
  const body = await request
    .clone()
    .json()
    .catch(() => undefined)
  if (typeof body !== "object" || body === null || !("method" in body) || typeof body.method !== "string") {
    return request.method
  }
  return body.method
}

function remote(url: string) {
  return { type: "remote" as const, url, oauth: false as const, timeout: 2_000 }
}
