import { expect } from "bun:test"
import { Effect } from "effect"
import { AISDK } from "@opencode-ai/core/aisdk"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { testEffect } from "./lib/effect"

const it = testEffect(AISDK.locationLayer)
const encoder = new TextEncoder()

function wrappedResponse(body: ReadableStream<Uint8Array>, signals: AbortSignal[]) {
  return Effect.gen(function* () {
    const aisdk = yield* AISDK.Service
    const fetches: ((input: Parameters<typeof fetch>[0], init?: RequestInit) => Promise<Response>)[] = []
    yield* aisdk.hook.sdk((event) => {
      fetches.push(event.options.fetch)
      event.sdk = { languageModel: () => ({}) }
    })
    yield* aisdk.language(
      ModelV2.Info.make({
        ...ModelV2.Info.empty(ProviderV2.ID.make("test"), ModelV2.ID.make("test-model")),
        api: {
          id: ModelV2.ID.make("test-model"),
          type: "aisdk",
          package: "test-provider",
          settings: {
            chunkTimeout: 70,
            fetch: (_: Parameters<typeof fetch>[0], init?: RequestInit) => {
              if (init?.signal) signals.push(init.signal)
              return Promise.resolve(new Response(body, { headers: { "content-type": "text/event-stream" } }))
            },
          },
        },
      }),
    )
    const wrapped = fetches.at(0)
    if (!wrapped) throw new Error("AISDK hook did not receive wrapped fetch")
    return yield* Effect.promise(() => wrapped("https://example.test"))
  })
}

for (const ending of ["\r", "\r\n"] as const) {
  it.live(`SSE fields with ${JSON.stringify(ending)} line endings keep the timeout alive across chunks`, () =>
    Effect.gen(function* () {
      const chunks = Array.from({ length: 8 }, (_, i) =>
        ending === "\r\n" ? [`data: ${i}\r`, "\n"] : [`data: ${i}\r`],
      ).flat()
      const expected = chunks.join("")
      const body = new ReadableStream<Uint8Array>({
        async pull(ctrl) {
          await Bun.sleep(15)
          const chunk = chunks.shift()
          if (chunk === undefined) return ctrl.close()
          ctrl.enqueue(encoder.encode(chunk))
        },
      })
      const response = yield* wrappedResponse(body, [])
      const reader = response.body?.getReader()
      if (!reader) throw new Error("missing SSE body")
      const result = yield* Effect.promise(async () => {
        const chunks: string[] = []
        for (;;) {
          const part = await reader.read()
          if (part.done) return chunks.join("")
          chunks.push(new TextDecoder().decode(part.value))
        }
      })
      expect(result).toBe(expected)
    }),
  )
}

it.live("comment-only SSE traffic does not reset the timeout", () =>
  Effect.gen(function* () {
    const signals: AbortSignal[] = []
    const response = yield* wrappedResponse(
      new ReadableStream<Uint8Array>({
        async pull(ctrl) {
          await Bun.sleep(10)
          ctrl.enqueue(encoder.encode(": keepalive\r\n"))
        },
      }),
      signals,
    )
    const reader = response.body?.getReader()
    if (!reader) throw new Error("missing SSE body")
    expect((yield* Effect.promise(() => reader.read())).done).toBe(false)
    yield* Effect.promise(() => Bun.sleep(120))
    expect(signals[0]?.aborted).toBe(false)
    const error = yield* Effect.promise(async () => {
      try {
        for (;;) {
          if ((await reader.read()).done) throw new Error("SSE closed without timeout error")
        }
      } catch (error) {
        return error
      }
    })
    expect(error).toBeInstanceOf(Error)
    if (!(error instanceof Error)) throw error
    expect(error.message).toBe("SSE read timed out")
  }),
)

it.live("consumer backpressure does not time out or discard unread SSE chunks", () =>
  Effect.gen(function* () {
    const signals: AbortSignal[] = []
    const response = yield* wrappedResponse(
      new ReadableStream<Uint8Array>({
        start(ctrl) {
          for (const chunk of ["data: one\n\n", "data: two\n\n", "data: three\n\n"]) ctrl.enqueue(encoder.encode(chunk))
          ctrl.close()
        },
      }),
      signals,
    )
    const reader = response.body?.getReader()
    if (!reader) throw new Error("missing SSE body")
    expect(new TextDecoder().decode((yield* Effect.promise(() => reader.read())).value)).toBe("data: one\n\n")
    yield* Effect.promise(() => Bun.sleep(120))
    expect(signals[0]?.aborted).toBe(false)
    expect(new TextDecoder().decode((yield* Effect.promise(() => reader.read())).value)).toBe("data: two\n\n")
    expect(new TextDecoder().decode((yield* Effect.promise(() => reader.read())).value)).toBe("data: three\n\n")
    expect((yield* Effect.promise(() => reader.read())).done).toBe(true)
  }),
)

for (const failure of [new Error("upstream failed"), new DOMException("stopped", "AbortError")]) {
  it.live(`reader failure ${failure.name} clears the timer without changing the error`, () =>
    Effect.gen(function* () {
      const signals: AbortSignal[] = []
      const response = yield* wrappedResponse(
        new ReadableStream<Uint8Array>({
          pull(ctrl) {
            ctrl.error(failure)
          },
        }),
        signals,
      )
      const reader = response.body?.getReader()
      if (!reader) throw new Error("missing SSE body")
      const error = yield* Effect.promise(async () => {
        try {
          await reader.read()
        } catch (error) {
          return error
        }
      })
      expect(error).toBe(failure)
      yield* Effect.promise(() => Bun.sleep(120))
      expect(signals[0]?.aborted).toBe(false)
    }),
  )
}
