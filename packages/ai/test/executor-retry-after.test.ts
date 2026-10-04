import { expect, test } from "bun:test"
import { Effect } from "effect"
import { HttpClientRequest } from "effect/unstable/http"
import { RequestExecutor } from "../src/route.js"

// Exercise the native fetch/status path, not only the hint parser.
test("native HTTP retry hints obey unsigned grammar and authoritative zero precedence", async () => {
  const cases: Array<{ headers: Record<string, string>; expected: number | undefined }> = [
    { headers: { "retry-after-ms": "0", "retry-after": "9" }, expected: 0 },
    { headers: { "retry-after-ms": " 0.5 ", "retry-after": "9" }, expected: 0.5 },
    { headers: { "retry-after": "2" }, expected: 2000 },
    { headers: { "retry-after": "0" }, expected: 0 },
    ...["", " ", "-1", "+1", "1e2", "0x10", "Infinity", ".5", "1."].map((value) => ({
      headers: { "retry-after-ms": value, "retry-after": "3" },
      expected: 3000,
    })),
    ...["", " ", "-1", "+1", "1.5", "1e2", "0x10", "January 1, 2099", "2099-01-01"].map((value) => ({
      headers: { "retry-after": value },
      expected: undefined,
    })),
    { headers: { "retry-after": "Sun, 31 Nov 2099 00:00:00 GMT" }, expected: undefined },
    { headers: { "retry-after": "Sun, 01 Jan 2099 00:00:00 GMT" }, expected: undefined },
    { headers: { "retry-after": "Sun, 06 Nov 1994 08:49:37 GMT" }, expected: 0 },
    { headers: { "retry-after": "Sunday, 06-Nov-94 08:49:37 GMT" }, expected: 0 },
    { headers: { "retry-after": "Sun Nov  6 08:49:37 1994" }, expected: 0 },
  ]
  let calls = 0
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(input) {
      calls++
      const item = cases[Number(new URL(input.url).pathname.slice(1))]!
      return new Response('{"error":{"message":"rate limited"}}', { status: 429, headers: item.headers })
    },
  })
  try {
    for (const [index, item] of cases.entries()) {
      const error = await Effect.runPromise(
        Effect.gen(function* () {
          const executor = yield* RequestExecutor.Service
          return yield* executor
            .execute(HttpClientRequest.get(new URL(String(index), server.url).href))
            .pipe(Effect.flip)
        }).pipe(Effect.provide(RequestExecutor.fetchLayer), Effect.scoped),
      )
      expect(error.reason._tag).toBe("RateLimit")
      if (error.reason._tag !== "RateLimit") throw error
      expect(error.reason.retryAfterMs, JSON.stringify(item.headers)).toBe(item.expected)
      expect(error.reason.http?.status).toBe(429)
      expect(error.reason.body).toBe('{"error":{"message":"rate limited"}}')
    }
    expect(calls).toBe(cases.length)
  } finally {
    server.stop(true)
  }
})
