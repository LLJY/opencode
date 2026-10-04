import { expect, test } from "bun:test"
import { Effect, Stream } from "effect"
import { FetchHttpClient } from "effect/unstable/http"
import { HttpOptions, LLM, mergeHttpOptions } from "../src/index.js"
import { LLMClient } from "../src/route.js"
import { route } from "../src/protocols/openai-chat.js"
import { runtimeLayer } from "./lib/http.js"
import { deltaChunk } from "./lib/openai-chunks.js"

test("native HTTP whole-request timeout bounds waiting for response headers", async () => {
  let timer: ReturnType<typeof setTimeout> | undefined
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: () =>
      new Promise<Response>((resolve) => {
        timer = setTimeout(() => resolve(new Response("{}")), 250)
      }),
  })
  try {
    const model = route.with({ endpoint: { baseURL: server.url.href } }).model({ id: "test-model" })
    const result = await Effect.runPromise(
      LLMClient.generate(LLM.request({ model, prompt: "Hello", http: { timeout: 60 } })).pipe(
        Effect.result,
        Effect.provide(runtimeLayer(FetchHttpClient.layer)),
        Effect.scoped,
      ),
    )
    expect(result._tag).toBe("Failure")
    if (result._tag !== "Failure") throw new Error("expected request timeout")
    expect(result.failure.reason).toMatchObject({ _tag: "Transport", code: "Timeout", operation: "request" })
    expect(result.failure.reason.http).toBeUndefined()
  } finally {
    clearTimeout(timer)
    server.stop(true)
  }
})

test("native HTTP whole-request deadline is not renewed by consumer pauses", async () => {
  let timer: ReturnType<typeof setTimeout> | undefined
  const chunk = (text: string, terminal = false) =>
    `data: ${JSON.stringify({ ...deltaChunk({ content: text }), choices: [{ index: 0, delta: { content: text }, finish_reason: terminal ? "stop" : null }] })}\n\n${terminal ? "data: [DONE]\n\n" : ""}`
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch: () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(chunk("one")))
            timer = setTimeout(() => {
              controller.enqueue(new TextEncoder().encode(chunk("two", true)))
              controller.close()
            }, 30)
          },
          cancel() {
            clearTimeout(timer)
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      ),
  })
  try {
    const model = route.with({ endpoint: { baseURL: server.url.href } }).model({ id: "test-model" })
    const result = await Effect.runPromise(
      LLMClient.stream(LLM.request({ model, prompt: "Hello", http: { timeout: 60, chunkTimeout: 60 } })).pipe(
        Stream.runForEach((event) => (event.type === "text-delta" ? Effect.sleep(120) : Effect.void)),
        Effect.result,
        Effect.provide(runtimeLayer(FetchHttpClient.layer)),
        Effect.scoped,
      ),
    )
    expect(result._tag).toBe("Failure")
    if (result._tag !== "Failure") throw new Error("expected elapsed deadline")
    expect(result.failure.reason).toMatchObject({
      _tag: "Transport",
      code: "Timeout",
      operation: "read",
      http: { status: 200 },
    })
  } finally {
    clearTimeout(timer)
    server.stop(true)
  }
})

test("HTTP options preserve timeout overlays and explicit disabling", () => {
  const merged = mergeHttpOptions(
    new HttpOptions({ timeout: 100, chunkTimeout: 50, headers: { base: "kept" }, body: { nested: { base: true } } }),
    new HttpOptions({ timeout: false, headers: { overlay: "kept" }, query: { page: "1" } }),
  )
  expect(merged).toMatchObject({
    timeout: false,
    chunkTimeout: 50,
    headers: { base: "kept", overlay: "kept" },
    body: { nested: { base: true } },
    query: { page: "1" },
  })
})
