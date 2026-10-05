import { expect } from "bun:test"
import { Effect, Stream } from "effect"
import { TestClock } from "effect/testing"
import { LLM } from "../src/index.js"
import { LLMClient } from "../src/route.js"
import { route } from "../src/protocols/openai-chat.js"
import { testEffect } from "./lib/effect.js"
import { fixedResponse } from "./lib/http.js"
import { deltaChunk, finishChunk, toolCallChunk } from "./lib/openai-chunks.js"

const model = route.model({ id: "buffered-deadline" })
const frame = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`

for (const fixture of [
  { name: "text", tail: frame(deltaChunk({ content: "late" })) + frame(finishChunk("stop")) },
  { name: "tool call", tail: frame(toolCallChunk("late-call", "effect", "{}")) + frame(finishChunk("tool_calls")) },
  { name: "terminal frame", tail: frame(finishChunk("stop")) },
  // A single protocol frame can emit several native events, including completion.
  { name: "terminal event from the same frame", tail: "", finish: "stop" },
] as const) {
  const body =
    frame(deltaChunk({ content: "first" }, "finish" in fixture ? fixture.finish : null)) +
    fixture.tail +
    "data: [DONE]\n\n"
  const it = testEffect(fixedResponse(body))
  it.effect(`native HTTP absolute deadline stops buffered ${fixture.name}`, () =>
    Effect.gen(function* () {
      const seen: string[] = []
      const result = yield* LLMClient.stream(
        LLM.request({ model, prompt: "Hello", http: { timeout: 60, chunkTimeout: 30 } }),
      ).pipe(
        Stream.runForEach((event) =>
          Effect.gen(function* () {
            seen.push(event.type)
            if (event.type === "text-delta") yield* TestClock.adjust(120)
          }),
        ),
        Effect.result,
      )
      expect(result._tag).toBe("Failure")
      if (result._tag !== "Failure") throw new Error("buffered events bypassed the absolute deadline")
      expect(result.failure.reason).toMatchObject({
        _tag: "Transport",
        code: "Timeout",
        operation: "read",
        http: { status: 200 },
      })
      expect(seen.filter((type) => type === "text-delta")).toHaveLength(1)
      expect(seen).not.toContain("tool-call")
      expect(seen).not.toContain("step-finish")
    }),
  )
}

const terminal = testEffect(
  fixedResponse(
    frame(toolCallChunk("accepted-call", "effect", "{}")) + frame(finishChunk("tool_calls")) + "data: [DONE]\n\n",
  ),
)
terminal.effect("native HTTP absolute deadline also checks native completion buffered after DONE", () =>
  Effect.gen(function* () {
    const seen: string[] = []
    const result = yield* LLMClient.stream(LLM.request({ model, prompt: "Hello", http: { timeout: 60 } })).pipe(
      Stream.runForEach((event) =>
        Effect.gen(function* () {
          seen.push(event.type)
          if (event.type === "tool-call") yield* TestClock.adjust(120)
        }),
      ),
      Effect.result,
    )
    expect(result._tag).toBe("Failure")
    if (result._tag !== "Failure") throw new Error("native completion bypassed the absolute deadline")
    expect(result.failure.reason).toMatchObject({ _tag: "Transport", code: "Timeout", operation: "read" })
    expect(seen).toContain("tool-call")
    expect(seen).not.toContain("step-finish")
  }),
)

for (const timeout of [undefined, false, 500] as const) {
  const it = testEffect(
    fixedResponse(
      frame(deltaChunk({ content: "first" })) +
        frame(deltaChunk({ content: "second" })) +
        frame(finishChunk("stop")) +
        "data: [DONE]\n\n",
    ),
  )
  it.effect(`native HTTP buffered progress excludes consumer pause from chunk wait (total ${timeout})`, () =>
    Effect.gen(function* () {
      const seen: string[] = []
      yield* LLMClient.stream(LLM.request({ model, prompt: "Hello", http: { timeout, chunkTimeout: 30 } })).pipe(
        Stream.runForEach((event) =>
          Effect.gen(function* () {
            seen.push(event.type)
            if (event.type === "text-delta") yield* TestClock.adjust(120)
          }),
        ),
      )
      expect(seen.filter((type) => type === "text-delta")).toHaveLength(2)
      expect(seen).toContain("step-finish")
    }),
  )
}
