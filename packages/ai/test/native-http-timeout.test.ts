import { expect, test } from "bun:test"
import { Cause, Effect, Stream } from "effect"
import { FetchHttpClient } from "effect/unstable/http"
import { LLM } from "../src/index.js"
import { LLMClient } from "../src/route.js"
import { route } from "../src/protocols/openai-chat.js"
import { runtimeLayer } from "./lib/http.js"
import { deltaChunk } from "./lib/openai-chunks.js"

test("native HTTP SSE chunk timeout ignores comment heartbeats", async () => {
  let cancelled = false
  const timers: Array<ReturnType<typeof setTimeout>> = []
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch() {
      return new Response(
        new ReadableStream({
          start(controller) {
            const encoder = new TextEncoder()
            controller.enqueue(encoder.encode(": ready\r\n\r\n"))
            for (let i = 1; i < 15; i++)
              timers.push(
                setTimeout(() => {
                  if (!cancelled) controller.enqueue(encoder.encode(": heartbeat\r\n\r\n"))
                }, i * 15),
              )
            timers.push(
              setTimeout(() => {
                if (cancelled) return
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify({ ...deltaChunk({ content: "Hello" }), choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`,
                  ),
                )
                controller.close()
              }, 250),
            )
          },
          cancel() {
            cancelled = true
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      )
    },
  })
  try {
    const model = route.with({ endpoint: { baseURL: server.url.href } }).model({ id: "test-model" })
    const result = await Effect.runPromise(
      LLMClient.generate(LLM.request({ model, prompt: "Hello", http: { chunkTimeout: 60 } })).pipe(
        Effect.result,
        Effect.provide(runtimeLayer(FetchHttpClient.layer)),
        Effect.scoped,
      ),
    )
    expect(result._tag).toBe("Failure")
    if (result._tag !== "Failure") throw new Error("expected timeout")
    expect(result.failure.reason).toMatchObject({
      _tag: "Transport",
      code: "Timeout",
      operation: "read",
      http: { status: 200 },
    })
  } finally {
    cancelled = true
    timers.forEach(clearTimeout)
    server.stop(true)
  }
})

test("native HTTP SSE does not charge consumer pauses or discard queued frames", async () => {
  const timers: Array<ReturnType<typeof setTimeout>> = []
  const chunk = (text: string, terminal = false) =>
    `data: ${JSON.stringify({
      ...deltaChunk({ content: text }),
      choices: [{ index: 0, delta: { content: text }, finish_reason: terminal ? "stop" : null }],
    })}\n\n${terminal ? "data: [DONE]\n\n" : ""}`
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch() {
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(chunk("one")))
            timers.push(
              setTimeout(() => {
                controller.enqueue(new TextEncoder().encode(chunk("two", true)))
                controller.close()
              }, 90),
            )
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      )
    },
  })
  let deltas = 0
  try {
    const model = route.with({ endpoint: { baseURL: server.url.href } }).model({ id: "test-model" })
    await Effect.runPromise(
      LLMClient.stream(LLM.request({ model, prompt: "Hello", http: { chunkTimeout: 60 } })).pipe(
        Stream.runForEach((event) => {
          if (event.type !== "text-delta") return Effect.void
          deltas++
          return deltas === 1 ? Effect.sleep(180) : Effect.void
        }),
        Effect.provide(runtimeLayer(FetchHttpClient.layer)),
        Effect.scoped,
      ),
    )
    expect(deltas).toBe(2)
  } finally {
    timers.forEach(clearTimeout)
    server.stop(true)
  }
})

test("native HTTP SSE caller cancellation stays interruption, not a timeout", async () => {
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch() {
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(": ready\n\n"))
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      )
    },
  })
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 40)
  try {
    const model = route.with({ endpoint: { baseURL: server.url.href } }).model({ id: "test-model" })
    const exit = await Effect.runPromiseExit(
      LLMClient.generate(LLM.request({ model, prompt: "Hello", http: { chunkTimeout: 500 } })).pipe(
        Effect.provide(runtimeLayer(FetchHttpClient.layer)),
        Effect.scoped,
      ),
      { signal: controller.signal },
    )
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") expect(Cause.hasInterrupts(exit.cause)).toBe(true)
  } finally {
    clearTimeout(timer)
    server.stop(true)
  }
})
