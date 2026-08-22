import { describe, expect } from "bun:test"
import { Effect, Fiber, Layer, Random, Ref } from "effect"
import * as TestClock from "effect/testing/TestClock"
import { Headers, HttpClient, HttpClientRequest, HttpClientResponse } from "effect/unstable/http"
import { LLM, LLMError } from "../src"
import { LLMClient, RequestExecutor } from "../src/route"
import * as OpenAIChat from "../src/protocols/openai-chat"
import { dynamicResponse } from "./lib/http"
import { deltaChunk } from "./lib/openai-chunks"
import { sseRaw } from "./lib/sse"
import { it } from "./lib/effect"

const request = HttpClientRequest.post("https://provider.test/v1/chat?api_key=secret&key=secret&debug=1").pipe(
  HttpClientRequest.setHeaders(Headers.fromInput({ authorization: "Bearer secret", "x-safe": "visible" })),
)

const secretRequest = HttpClientRequest.post("https://provider.test/v1/chat?api_key=query-secret-123&debug=1").pipe(
  HttpClientRequest.setHeaders(Headers.fromInput({ authorization: "Bearer header-secret-456" })),
)

const escapedSecretRequest = HttpClientRequest.post("https://provider.test/v1/chat").pipe(
  HttpClientRequest.setHeaders(Headers.fromInput({ authorization: 'Bearer he"llo-secret-123' })),
)

const responsesLayer = (responses: ReadonlyArray<Response>) =>
  RequestExecutor.layer.pipe(
    Layer.provide(
      Layer.unwrap(
        Effect.gen(function* () {
          const cursor = yield* Ref.make(0)
          return Layer.succeed(
            HttpClient.HttpClient,
            HttpClient.make((request) =>
              Effect.gen(function* () {
                const index = yield* Ref.getAndUpdate(cursor, (value) => value + 1)
                return HttpClientResponse.fromWeb(request, responses[index] ?? responses[responses.length - 1])
              }),
            ),
          )
        }),
      ),
    ),
  )

const countedResponsesLayer = (attempts: Ref.Ref<number>, responses: ReadonlyArray<Response>) =>
  RequestExecutor.layer.pipe(
    Layer.provide(
      Layer.unwrap(
        Effect.gen(function* () {
          const cursor = yield* Ref.make(0)
          return Layer.succeed(
            HttpClient.HttpClient,
            HttpClient.make((request) =>
              Effect.gen(function* () {
                yield* Ref.update(attempts, (value) => value + 1)
                const index = yield* Ref.getAndUpdate(cursor, (value) => value + 1)
                return HttpClientResponse.fromWeb(request, responses[index] ?? responses[responses.length - 1])
              }),
            ),
          )
        }),
      ),
    ),
  )

const randomMidpoint = {
  nextDoubleUnsafe: () => 0.5,
  nextIntUnsafe: () => 0,
}

const FAR_FUTURE_HTTP_DATES = ["Fri, 31 Dec 2100 23:59:59 GMT", "Fri Dec 31 23:59:59 2100"]
const MALFORMED_RETRY_AFTER_DATES = [
  "January 1, 2099",
  "Tomorrow",
  "sun, 06 nov 1994 08:49:37 gmt",
  "2099-01-01T00:00:00Z",
]

// `Number` accepts all of these; the header grammars do not.
const NON_DECIMAL_VALUES = ["+5", "-5", "1e3", "Infinity", "0x10", "", " ", "5ms", ".5"]

// RFC 9110 delta-seconds is `1*DIGIT`, so the standard header additionally rejects
// the fractions the proprietary millisecond header still accepts.
const NON_DELTA_SECONDS_VALUES = [...NON_DECIMAL_VALUES, "1.5", "2.0", "0.5"]

// Drives one 503 -> 200 retry and asserts the retry fires exactly at `delay`.
const retryDelayCase = (headers: Record<string, string>, delay: number) =>
  Effect.gen(function* () {
    const attempts = yield* Ref.make(0)
    return yield* Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const fiber = yield* executor.execute(request).pipe(Effect.forkChild)

      yield* Effect.yieldNow
      expect(yield* Ref.get(attempts)).toBe(1)

      yield* TestClock.adjust(delay - 1)
      yield* Effect.yieldNow
      expect(yield* Ref.get(attempts)).toBe(1)

      yield* TestClock.adjust(1)
      const response = yield* Fiber.join(fiber)

      expect(response.status).toBe(200)
      expect(yield* Ref.get(attempts)).toBe(2)
    }).pipe(
      Effect.provide(
        countedResponsesLayer(attempts, [
          new Response("busy", { status: 503, headers }),
          new Response("ok", { status: 200 }),
        ]),
      ),
    )
  })

const expectLLMError = (error: unknown) => {
  expect(error).toBeInstanceOf(LLMError)
  if (!(error instanceof LLMError)) throw new Error("expected LLMError")
  return error
}

const errorHttp = (error: LLMError) => ("http" in error.reason ? error.reason.http : undefined)

describe("RequestExecutor", () => {
  it.effect("classifies context overflow responses", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(error.reason).toMatchObject({ _tag: "InvalidRequest", classification: "context-overflow" })
    }).pipe(
      Effect.provide(
        responsesLayer([
          new Response('{"error":{"code":"context_length_exceeded","message":"prompt too long"}}', {
            status: 400,
          }),
        ]),
      ),
    ),
  )

  it.effect("does not classify generic HTTP 413 payload errors as context overflow", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(error.reason).toMatchObject({ _tag: "InvalidRequest" })
      expect("classification" in error.reason ? error.reason.classification : undefined).toBeUndefined()
    }).pipe(Effect.provide(responsesLayer([new Response("request too large", { status: 413 })]))),
  )

  it.effect("does not classify ordinary invalid requests as context overflow", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(error.reason).toMatchObject({ _tag: "InvalidRequest" })
      expect("classification" in error.reason ? error.reason.classification : undefined).toBeUndefined()
    }).pipe(Effect.provide(responsesLayer([new Response("invalid parameter", { status: 400 })]))),
  )

  it.effect("returns redacted diagnostics for retryable rate limits", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(error).toMatchObject({
        retryable: true,
        retryAfterMs: 0,
        reason: {
          _tag: "RateLimit",
          rateLimit: { retryAfterMs: 0 },
          http: {
            requestId: "req_123",
            request: {
              method: "POST",
              url: "https://provider.test/v1/chat?api_key=%3Credacted%3E&key=%3Credacted%3E&debug=1",
              headers: { authorization: "<redacted>", "x-safe": "visible" },
            },
            response: {
              status: 429,
              headers: {
                "retry-after-ms": "0",
                "x-request-id": "req_123",
                "x-api-key": "<redacted>",
              },
            },
          },
        },
      })
      expect(errorHttp(error)?.body).toBe("rate limited")
    }).pipe(
      Effect.provide(
        responsesLayer(
          Array.from(
            { length: 3 },
            () =>
              new Response("rate limited", {
                status: 429,
                headers: { "retry-after-ms": "0", "x-request-id": "req_123", "x-api-key": "secret" },
              }),
          ),
        ),
      ),
    ),
  )

  it.effect("honors current redacted header names in diagnostics", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.request.headers["x-safe"]).toBe("<redacted>")
      expect(errorHttp(error)?.response?.headers["x-safe"]).toBe("<redacted>")
    }).pipe(
      Effect.provide(responsesLayer([new Response("bad", { status: 400, headers: { "x-safe": "response-secret" } })])),
      Effect.provideService(Headers.CurrentRedactedNames, ["x-safe"]),
    ),
  )

  it.effect("extracts OpenAI-style rate-limit diagnostics", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(error.reason).toMatchObject({ _tag: "RateLimit" })
      expect(error.reason._tag === "RateLimit" ? error.reason.rateLimit : undefined).toEqual({
        retryAfterMs: 0,
        limit: { requests: "500", tokens: "30000" },
        remaining: { requests: "499", tokens: "29900" },
        reset: { requests: "1s", tokens: "10s" },
      })
    }).pipe(
      Effect.provide(
        responsesLayer(
          Array.from(
            { length: 3 },
            () =>
              new Response("rate limited", {
                status: 429,
                headers: {
                  "retry-after-ms": "0",
                  "x-ratelimit-limit-requests": "500",
                  "x-ratelimit-limit-tokens": "30000",
                  "x-ratelimit-remaining-requests": "499",
                  "x-ratelimit-remaining-tokens": "29900",
                  "x-ratelimit-reset-requests": "1s",
                  "x-ratelimit-reset-tokens": "10s",
                },
              }),
          ),
        ),
      ),
    ),
  )

  it.effect("extracts Anthropic-style rate-limit diagnostics", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(error.reason).toMatchObject({ _tag: "ProviderInternal" })
      expect(errorHttp(error)?.rateLimit).toEqual({
        retryAfterMs: 0,
        limit: { requests: "100", "input-tokens": "10000" },
        remaining: { requests: "12", "input-tokens": "9000" },
        reset: { requests: "2026-05-06T12:00:00Z", "input-tokens": "2026-05-06T12:00:10Z" },
      })
    }).pipe(
      Effect.provide(
        responsesLayer(
          Array.from(
            { length: 3 },
            () =>
              new Response("overloaded", {
                status: 529,
                headers: {
                  "retry-after-ms": "0",
                  "anthropic-ratelimit-requests-limit": "100",
                  "anthropic-ratelimit-requests-remaining": "12",
                  "anthropic-ratelimit-requests-reset": "2026-05-06T12:00:00Z",
                  "anthropic-ratelimit-input-tokens-limit": "10000",
                  "anthropic-ratelimit-input-tokens-remaining": "9000",
                  "anthropic-ratelimit-input-tokens-reset": "2026-05-06T12:00:10Z",
                },
              }),
          ),
        ),
      ),
    ),
  )

  it.effect("retries retryable status responses before returning the stream", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const response = yield* executor.execute(request)

      expect(response.status).toBe(200)
      expect(yield* response.text).toBe("ok")
    }).pipe(
      Effect.provide(
        responsesLayer([
          new Response("busy", { status: 503, headers: { "retry-after-ms": "0" } }),
          new Response("ok", { status: 200 }),
        ]),
      ),
    ),
  )

  it.effect("omits HTML error bodies from user-facing provider messages", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(error.reason).toMatchObject({
        _tag: "ProviderInternal",
        status: 503,
        message: "Provider request failed with HTTP 503",
      })
      expect(error.reason.message).not.toMatch(/<html/i)
      expect(error.reason.message).not.toMatch(/nginx/i)
    }).pipe(
      Effect.provide(
        responsesLayer([
          new Response(
            "<html>\n<head><title>503 Service Temporarily Unavailable</title></head>\n<body><center><h1>503 Service Temporarily Unavailable</h1></center><hr><center>nginx</center></body>\n</html>",
            { status: 503, headers: { "retry-after-ms": "0" } },
          ),
          new Response(
            "<html>\n<head><title>503 Service Temporarily Unavailable</title></head>\n<body><center><h1>503 Service Temporarily Unavailable</h1></center></body>\n</html>",
            { status: 503, headers: { "retry-after-ms": "0" } },
          ),
          new Response(
            "<html>\n<head><title>503 Service Temporarily Unavailable</title></head>\n<body><center><h1>503 Service Temporarily Unavailable</h1></center></body>\n</html>",
            { status: 503 },
          ),
        ]),
      ),
    ),
  )

  it.effect("marks 504 and 529 status responses retryable", () =>
    Effect.gen(function* () {
      const failWith = (status: number) =>
        Effect.gen(function* () {
          const executor = yield* RequestExecutor.Service
          const error = yield* executor.execute(request).pipe(Effect.flip)

          expectLLMError(error)
          expect(error.reason).toMatchObject({ _tag: "ProviderInternal", status })
          expect(error.retryable).toBe(true)
        }).pipe(
          Effect.provide(
            responsesLayer(
              Array.from(
                { length: 3 },
                () =>
                  new Response("retry", {
                    status,
                    headers: { "retry-after-ms": "0" },
                  }),
              ),
            ),
          ),
        )

      yield* failWith(504)
      yield* failWith(529)
    }),
  )

  it.effect("does not retry non-retryable status responses and truncates large bodies", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(error.reason).toMatchObject({ _tag: "Authentication" })
      expect(error.retryable).toBe(false)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      expect(errorHttp(error)?.body).toHaveLength(16_384)
    }).pipe(
      Effect.provide(
        responsesLayer([
          new Response("x".repeat(20_000), { status: 401 }),
          new Response("should not retry", { status: 200 }),
        ]),
      ),
    ),
  )

  it.effect("stops reading a large error body and cancels the rest of the stream", () => {
    let pulled = 0
    let cancelled = false
    const chunk = new TextEncoder().encode("x".repeat(4_096))
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulled++
        if (pulled > 64) return controller.close()
        controller.enqueue(chunk)
      },
      cancel() {
        cancelled = true
      },
    })

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      expect(errorHttp(error)?.body).toHaveLength(16_384)
      // The read budget is 20 KiB, so a handful of 4 KiB chunks ends it far short of
      // the 256 KiB the provider is still willing to send.
      expect(pulled).toBeLessThanOrEqual(8)
      // The abandoned read is signalled synchronously, so the cancel has already landed
      // by the time the diagnostic is built.
      yield* Effect.yieldNow
      expect(cancelled).toBe(true)
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 401 })])))
  })

  it.effect("decodes UTF-8 sequences split across error body chunks", () => {
    const bytes = new TextEncoder().encode('{"error":"héllo"}')
    const split = bytes.indexOf(0xc3) + 1
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, split))
        controller.enqueue(bytes.slice(split))
        controller.close()
      },
    })

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.body).toBe('{"error":"héllo"}')
      expect(errorHttp(error)?.bodyTruncated).toBeUndefined()
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 400 })])))
  })

  // The budget counts bytes off the wire, so a provider cannot spend it in one
  // allocation by sending the whole error page as a single frame.
  it.effect("bounds an error body delivered as one huge chunk", () => {
    let pulled = 0
    const huge = new TextEncoder().encode("x".repeat(1_048_576))
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulled++
        controller.enqueue(huge)
      },
    })

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      expect(errorHttp(error)?.body).toHaveLength(16_384)
      // One frame already covers the whole budget; the source only reads ahead by
      // its own high-water mark, never because the read asked for more.
      expect(pulled).toBeLessThanOrEqual(2)
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 401 })])))
  })

  // 20 KiB is not a multiple of three, so the byte budget lands two bytes into a
  // euro sign: decoding that tail would report a replacement character instead.
  it.effect("drops a multibyte sequence the byte budget cut in half", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      expect(errorHttp(error)?.body).not.toContain("\uFFFD")
      expect(errorHttp(error)?.body).toHaveLength(6_826)
    }).pipe(Effect.provide(responsesLayer([new Response("\u20AC".repeat(8_000), { status: 400 })]))),
  )

  it.effect("does not split a surrogate pair at the truncation limit", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      expect(errorHttp(error)?.body).toHaveLength(16_383)
      expect(errorHttp(error)?.body?.endsWith("x")).toBe(true)
      expect(errorHttp(error)?.body).not.toContain("\uD83D")
    }).pipe(
      Effect.provide(
        responsesLayer([new Response("x".repeat(16_383) + "\u{1F600}" + "y".repeat(5_000), { status: 400 })]),
      ),
    ),
  )

  it.effect("redacts a secret the provider echoed back JSON escaped", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(escapedSecretRequest).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.body).toContain('"api_key":"<redacted>"')
      expect(errorHttp(error)?.body).toContain("echoed <redacted>")
      expect(errorHttp(error)?.body).not.toContain("llo-secret-123")
    }).pipe(
      Effect.provide(
        responsesLayer([
          new Response('{"error":{"message":"echoed he\\"llo-secret-123"},"api_key":"tail\\"end"}', { status: 400 }),
        ]),
      ),
    ),
  )

  // Cancelling the unread remainder is the provider's promise to keep, so a cancel
  // that never settles must not hold the error path open.
  it.effect("reports the buffered error body when the remainder cancel never settles", () => {
    const chunk = new TextEncoder().encode("x".repeat(4_096))
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(chunk)
      },
      cancel() {
        return new Promise<void>(() => {})
      },
    })

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      expect(errorHttp(error)?.body).toHaveLength(16_384)
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 401 })])))
  })

  // The bounded read overshoots the truncation limit precisely so redaction still
  // sees a whole secret that starts inside the budget and ends past it.
  it.effect("redacts a secret straddling the truncation limit", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(secretRequest).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      expect(errorHttp(error)?.body).not.toContain("query-secret")
      expect(errorHttp(error)?.body).toContain("<redact")
    }).pipe(
      Effect.provide(
        responsesLayer([new Response("x".repeat(16_376) + "query-secret-123" + "y".repeat(5_000), { status: 400 })]),
      ),
    ),
  )

  it.effect("redacts a partial request secret beyond the read overshoot", () => {
    const secret = `boundary-secret-${"a9f3".repeat(1_250)}`
    const longSecretRequest = HttpClientRequest.post("https://provider.test/v1/chat").pipe(
      HttpClientRequest.setHeaders(Headers.fromInput({ authorization: `Bearer ${secret}` })),
    )

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(longSecretRequest).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      // The secret starts inside the reported budget, so an unrepaired read leaks the
      // characters between where it starts and where the budget cuts the report.
      expect(errorHttp(error)?.body).not.toContain(secret.slice(0, 64))
      expect(errorHttp(error)?.body).toContain("<redacted>")
    }).pipe(Effect.provide(responsesLayer([new Response("x".repeat(16_000) + secret, { status: 400 })])))
  })

  it.effect("redacts an unterminated sensitive JSON field at the read limit", () => {
    // Deliberately not a value the literal pass would catch, so the assertion can
    // only pass because the field itself was recognized as sensitive.
    const value = `unterminated-${"a9f3".repeat(1_250)}`

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      expect(errorHttp(error)?.body).toContain('"api_key":"<redacted>"')
      expect(errorHttp(error)?.body).not.toContain(value.slice(0, 32))
    }).pipe(
      Effect.provide(responsesLayer([new Response("x".repeat(16_300) + `{"api_key":"${value}`, { status: 400 })])),
    )
  })

  // The provider truncated the secret itself, so the whole body fits inside the read
  // budget and arrives complete — but the report is still cut at BODY_LIMIT, and the
  // echoed prefix starts before that cut.
  it.effect("redacts a partial secret in a body that never reached the read limit", () => {
    const secret = `sk-boundary-${"a9f3".repeat(500)}`
    const boundaryRequest = HttpClientRequest.post("https://provider.test/v1/chat").pipe(
      HttpClientRequest.setHeaders(Headers.fromInput({ authorization: `Bearer ${secret}` })),
    )
    const body = "x".repeat(16_000) + secret.slice(0, 1_000)

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(boundaryRequest).pipe(Effect.flip)

      expectLLMError(error)
      expect(body.length).toBeLessThan(20_480)
      expect(errorHttp(error)?.body).not.toContain("sk-boundary-")
      expect(errorHttp(error)?.body?.endsWith("<redacted>")).toBe(true)
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 400 })])))
  })

  // The read can stop directly after a backslash, leaving an escape pair with nothing
  // to escape. The value is still sensitive.
  it.effect("redacts an unterminated sensitive JSON field that ends on a backslash", () => {
    const head = "x".repeat(16_000) + '{"api_key":"'
    const body = head + "b".repeat(20_480 - head.length - 1) + "\\"

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(body).toHaveLength(20_480)
      expect(errorHttp(error)?.body).toContain('"api_key":"<redacted>"')
      expect(errorHttp(error)?.body).not.toContain('"api_key":"bbbb')
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 400 })])))
  })

  // `.` stops at a line terminator, so an escaped newline would otherwise strand the
  // value pattern and report the whole sensitive value.
  it.effect("redacts an unterminated sensitive JSON field containing an escaped newline", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.body).toBe('{"api_key":"<redacted>"')
    }).pipe(Effect.provide(responsesLayer([new Response('{"api_key":"abcdefghij\\\n', { status: 400 })]))),
  )

  // The transport cut the body one byte into a two-byte sequence. Decoding that tail
  // would report a real character as U+FFFD, so the clipped sequence is dropped.
  it.effect("drops a multibyte sequence a failed body stream cut in half", () => {
    const bytes = new TextEncoder().encode("prefix héllo")
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, bytes.indexOf(0xc3) + 1))
      },
      pull(controller) {
        controller.error(new Error("connection reset"))
      },
    })

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.body).toBe("prefix h")
      expect(errorHttp(error)?.body).not.toContain("\uFFFD")
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 400 })])))
  })

  // Signalling the abandoned read must not create work that waits on the provider. A
  // cancel that never settles has to leave nothing running and nothing accumulating.
  it.live("abandons a never-settling cancel without retaining live read work", () => {
    let pulled = 0
    let cancelled = 0
    const chunk = new TextEncoder().encode("x".repeat(4_096))
    const body = () =>
      new ReadableStream<Uint8Array>({
        pull(controller) {
          pulled++
          controller.enqueue(chunk)
        },
        cancel() {
          cancelled++
          return new Promise<void>(() => {})
        },
      })

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service

      // Repeated so a teardown that parked work would accumulate it across attempts.
      for (const attempt of [1, 2, 3]) {
        const error = yield* executor.execute(request).pipe(Effect.flip, Effect.timeout("2 seconds"))
        expectLLMError(error)
        expect(errorHttp(error)?.body).toHaveLength(16_384)
        expect(cancelled).toBe(attempt)
      }

      // The reads are torn down, so nothing keeps pulling once the diagnostics are built.
      const settled = pulled
      yield* Effect.sleep("50 millis")
      expect(pulled).toBe(settled)
    }).pipe(
      Effect.provide(
        responsesLayer([
          new Response(body(), { status: 401 }),
          new Response(body(), { status: 401 }),
          new Response(body(), { status: 401 }),
        ]),
      ),
    )
  })

  // Signalling the abandoned read leaves it parked in the provider's own cancel, which
  // is the provider's to hold. What must not happen is the runtime holding it too: that
  // would strand one live read, and the buffer it filled, per hostile error for the rest
  // of the process. Nothing the diagnostic keeps refers back to the read, so once the
  // response is dropped the whole abandoned island is collectable.
  it.live("does not retain an abandoned read whose cancel never settles", () => {
    const sources: WeakRef<object>[] = []
    const chunk = new TextEncoder().encode("x".repeat(4_096))
    // Built per request and never held by the layer, so the only thing that could keep a
    // source alive is the read the executor abandoned.
    const lazyLayer = RequestExecutor.layer.pipe(
      Layer.provide(
        Layer.succeed(
          HttpClient.HttpClient,
          HttpClient.make((request) =>
            Effect.sync(() => {
              const source = {
                pull(controller: ReadableStreamDefaultController<Uint8Array>) {
                  controller.enqueue(chunk)
                },
                cancel() {
                  return new Promise<void>(() => {})
                },
              }
              sources.push(new WeakRef(source))
              return HttpClientResponse.fromWeb(
                request,
                new Response(new ReadableStream<Uint8Array>(source), { status: 401 }),
              )
            }),
          ),
        ),
      ),
    )

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service

      for (const _ of [1, 2, 3, 4, 5]) {
        const error = yield* executor.execute(request).pipe(Effect.flip, Effect.timeout("2 seconds"))
        expectLLMError(error)
        expect(errorHttp(error)?.body).toHaveLength(16_384)
      }

      yield* Effect.sleep("50 millis")
      Bun.gc(true)
      yield* Effect.sleep("10 millis")
      Bun.gc(true)

      expect(sources).toHaveLength(5)
      expect(sources.filter((ref) => ref.deref() !== undefined)).toHaveLength(0)
    }).pipe(Effect.provide(lazyLayer))
  })

  // Two secrets reach the tail: one only overlaps it by chance, the other is the one
  // actually cut. Repairing per secret would let the accidental overlap cut first and
  // hide the real match, leaving the head of the cut secret in the report.
  it.effect("cuts the longest secret tail rather than the first one that matches", () => {
    const echoed = "head-0123456789abcdef"
    const overlapping = "0123-xyz-token"
    const twoSecretRequest = HttpClientRequest.post("https://provider.test/v1/chat").pipe(
      HttpClientRequest.setHeaders(Headers.fromInput({ authorization: `Bearer ${overlapping}`, "x-api-key": echoed })),
    )
    // Three-byte filler keeps the read at its byte budget while leaving the decoded
    // text far under the reported budget, so the repaired tail is actually reported
    // instead of being sliced away.
    const body = "\u20AC".repeat(6_823) + "xx" + echoed

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(twoSecretRequest).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.bodyTruncated).toBe(true)
      // Asserted exactly rather than by absence: cutting at the first secret that
      // matches also removes `head-`, so only the full tail proves the longest won.
      expect(errorHttp(error)?.body).toBe("\u20AC".repeat(6_823) + "xx<redacted>")
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 400 })])))
  })

  // A connection reset leaves the same half-written secret behind as a budget stop, so
  // the repair cannot key off how much was read.
  it.effect("redacts a secret cut short by a failed body stream", () => {
    const secret = "sk-live-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    const cutRequest = HttpClientRequest.post("https://provider.test/v1/chat").pipe(
      HttpClientRequest.setHeaders(Headers.fromInput({ authorization: `Bearer ${secret}` })),
    )
    let sent = false
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (sent) return controller.error(new Error("connection reset"))
        sent = true
        controller.enqueue(new TextEncoder().encode(`diagnostics: ${secret.slice(0, 22)}`))
      },
    })

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(cutRequest).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.body).not.toContain("sk-live-")
      expect(errorHttp(error)?.body).toBe("diagnostics: <redacted>")
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 400 })])))
  })

  it.effect("redacts common secret fields in response bodies", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(request).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.body).toContain('"key":"<redacted>"')
      expect(errorHttp(error)?.body).toContain("api_key=<redacted>")
      expect(errorHttp(error)?.body).not.toContain("body-secret")
      expect(errorHttp(error)?.body).not.toContain("query-secret")
    }).pipe(
      Effect.provide(
        responsesLayer([
          new Response('{"error":{"message":"bad","key":"body-secret","detail":"api_key=query-secret"}}', {
            status: 400,
          }),
        ]),
      ),
    ),
  )

  it.effect("redacts echoed request secret values in response bodies", () =>
    Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const error = yield* executor.execute(secretRequest).pipe(Effect.flip)

      expectLLMError(error)
      expect(errorHttp(error)?.body).toContain("provider echoed <redacted>")
      expect(errorHttp(error)?.body).toContain("authorization <redacted>")
      expect(errorHttp(error)?.body).not.toContain("query-secret-123")
      expect(errorHttp(error)?.body).not.toContain("header-secret-456")
    }).pipe(
      Effect.provide(
        responsesLayer([
          new Response("provider echoed query-secret-123 and authorization header-secret-456", { status: 400 }),
        ]),
      ),
    ),
  )

  it.live("interrupts a stalled partial-body read", () => {
    const stalled = Promise.withResolvers<void>()
    const cancelled = Promise.withResolvers<void>()
    let sent = false
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (sent) {
          stalled.resolve()
          return new Promise<void>(() => {})
        }
        sent = true
        controller.enqueue(new TextEncoder().encode("partial"))
      },
      cancel() {
        cancelled.resolve()
        return new Promise<void>(() => {})
      },
    })

    return Effect.gen(function* () {
      const executor = yield* RequestExecutor.Service
      const fiber = yield* executor.execute(request).pipe(Effect.forkChild)

      yield* Effect.promise(() => stalled.promise).pipe(Effect.timeout("1 second"))
      yield* Fiber.interrupt(fiber).pipe(Effect.timeout("1 second"))
      yield* Effect.promise(() => cancelled.promise).pipe(Effect.timeout("1 second"))
    }).pipe(Effect.provide(responsesLayer([new Response(body, { status: 401 })])))
  })

  it.effect("honors Retry-After delta seconds before retrying", () =>
    Effect.gen(function* () {
      const attempts = yield* Ref.make(0)
      return yield* Effect.gen(function* () {
        const executor = yield* RequestExecutor.Service
        const fiber = yield* executor.execute(request).pipe(Effect.forkChild)

        yield* Effect.yieldNow
        expect(yield* Ref.get(attempts)).toBe(1)

        yield* TestClock.adjust(1_999)
        yield* Effect.yieldNow
        expect(yield* Ref.get(attempts)).toBe(1)

        yield* TestClock.adjust(1)
        const response = yield* Fiber.join(fiber)

        expect(response.status).toBe(200)
        expect(yield* Ref.get(attempts)).toBe(2)
      }).pipe(
        Effect.provide(
          countedResponsesLayer(attempts, [
            new Response("busy", { status: 503, headers: { "retry-after": "2" } }),
            new Response("ok", { status: 200 }),
          ]),
        ),
      )
    }),
  )

  // A valid HTTP-date is still honored; MAX_DELAY_MS keeps a far-future value bounded.
  FAR_FUTURE_HTTP_DATES.forEach((value) =>
    it.effect(`clamps far-future Retry-After HTTP-date "${value}" to the max delay`, () =>
      retryDelayCase({ "retry-after": value }, 10_000),
    ),
  )

  // Malformed dates must not turn into multi-week waits via permissive Date.parse.
  MALFORMED_RETRY_AFTER_DATES.forEach((value) =>
    it.effect(`falls back to jittered backoff for malformed Retry-After "${value}"`, () =>
      retryDelayCase({ "retry-after": value }, 500).pipe(Effect.provideService(Random.Random, randomMidpoint)),
    ),
  )

  // A malformed retry-after-ms must not short-circuit the lower-precedence header.
  NON_DECIMAL_VALUES.forEach((value) =>
    it.effect(`falls through to Retry-After when retry-after-ms is "${value}"`, () =>
      retryDelayCase({ "retry-after-ms": value, "retry-after": "2" }, 2_000),
    ),
  )

  NON_DELTA_SECONDS_VALUES.forEach((value) =>
    it.effect(`falls back to jittered backoff when Retry-After is "${value}"`, () =>
      retryDelayCase({ "retry-after": value }, 500).pipe(Effect.provideService(Random.Random, randomMidpoint)),
    ),
  )

  it.effect("accepts surrounding whitespace in retry hints", () =>
    Effect.gen(function* () {
      yield* retryDelayCase({ "retry-after-ms": " 250 " }, 250)
      yield* retryDelayCase({ "retry-after": " 2 " }, 2_000)
    }),
  )

  // `retry-after-ms` is proprietary and carries no delta-seconds grammar, so a
  // fraction there stays a hint and still outranks a well-formed Retry-After.
  it.effect("keeps honoring fractional retry-after-ms", () =>
    retryDelayCase({ "retry-after-ms": " 250.5 ", "retry-after": "2" }, 250.5),
  )

  it.effect("uses exponential jittered delay when retry-after is absent", () =>
    Effect.gen(function* () {
      const attempts = yield* Ref.make(0)
      return yield* Effect.gen(function* () {
        const executor = yield* RequestExecutor.Service
        const fiber = yield* executor.execute(request).pipe(Effect.flip, Effect.forkChild)

        yield* Effect.yieldNow
        expect(yield* Ref.get(attempts)).toBe(1)

        yield* TestClock.adjust(499)
        yield* Effect.yieldNow
        expect(yield* Ref.get(attempts)).toBe(1)

        yield* TestClock.adjust(1)
        yield* Effect.yieldNow
        expect(yield* Ref.get(attempts)).toBe(2)

        yield* TestClock.adjust(999)
        yield* Effect.yieldNow
        expect(yield* Ref.get(attempts)).toBe(2)

        yield* TestClock.adjust(1)
        const error = yield* Fiber.join(fiber)

        expectLLMError(error)
        expect(error.reason).toMatchObject({ _tag: "ProviderInternal" })
        expect(yield* Ref.get(attempts)).toBe(3)
      }).pipe(
        Effect.provide(
          countedResponsesLayer(attempts, [
            new Response("busy", { status: 503 }),
            new Response("still busy", { status: 503 }),
            new Response("done retrying", { status: 503 }),
          ]),
        ),
      )
    }).pipe(Effect.provideService(Random.Random, randomMidpoint)),
  )

  it.effect("does not retry after a successful response reaches stream parsing", () =>
    Effect.gen(function* () {
      const attempts = yield* Ref.make(0)
      const model = OpenAIChat.route
        .with({ endpoint: { baseURL: "https://api.openai.test/v1" } })
        .model({ id: "gpt-4o-mini" })
      const error = yield* LLMClient.generate(LLM.request({ model, prompt: "Say hello." })).pipe(
        Effect.provide(
          dynamicResponse((input) =>
            Ref.update(attempts, (value) => value + 1).pipe(
              Effect.as(
                input.respond(
                  sseRaw(
                    `data: ${JSON.stringify(deltaChunk({ role: "assistant", content: "Hello" }))}`,
                    "data: not-json",
                  ),
                  { headers: { "content-type": "text/event-stream" } },
                ),
              ),
            ),
          ),
        ),
        Effect.flip,
      )

      expectLLMError(error)
      expect(error.reason).toMatchObject({ _tag: "InvalidProviderOutput" })
      expect(yield* Ref.get(attempts)).toBe(1)
    }),
  )
})
