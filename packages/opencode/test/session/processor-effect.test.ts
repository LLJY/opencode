import { SessionV1 } from "@opencode-ai/core/v1/session"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { EventV2Bridge } from "@/event-v2-bridge"
import { expect } from "bun:test"
import { APICallError, JSONParseError, tool } from "ai"
import { InvalidProviderOutputReason, InvalidRequestReason, LLMError, LLMEvent } from "@opencode-ai/llm"
import { Cause, Effect, Exit, Fiber, Layer, Stream } from "effect"
import path from "path"
import { pathToFileURL } from "url"
import z from "zod"
import type { Agent } from "../../src/agent/agent"
import { ProviderError } from "../../src/provider/error"
import { Provider } from "@/provider/provider"

import { Session } from "@/session/session"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionProcessor } from "../../src/session/processor"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionStatus } from "../../src/session/status"
import { SessionSummary } from "../../src/session/summary"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { provideTmpdirInstance, provideTmpdirServer } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { raw, reply, TestLLMServer } from "../lib/llm-server"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { SessionProjector } from "@opencode-ai/core/session/projector"

const summary = Layer.succeed(
  SessionSummary.Service,
  SessionSummary.Service.of({
    summarize: () => Effect.void,
    diff: () => Effect.succeed([]),
    computeDiff: () => Effect.succeed([]),
  }),
)

const ref = {
  providerID: ProviderV2.ID.make("test"),
  modelID: ModelV2.ID.make("test-model"),
}

const cfg = {
  provider: {
    test: {
      name: "Test",
      id: "test",
      env: [],
      npm: "@ai-sdk/openai-compatible",
      models: {
        "test-model": {
          id: "test-model",
          name: "Test Model",
          attachment: false,
          reasoning: false,
          temperature: false,
          tool_call: true,
          release_date: "2025-01-01",
          limit: { context: 100000, output: 10000 },
          cost: { input: 0, output: 0 },
          options: {},
        },
      },
      options: {
        apiKey: "test-key",
        baseURL: "http://localhost:1/v1",
      },
    },
  },
}

function providerCfg(url: string) {
  return providerCfgWithOptions(url)
}

function providerCfgWithOptions(url: string, options?: Record<string, unknown>) {
  return {
    ...cfg,
    provider: {
      ...cfg.provider,
      test: {
        ...cfg.provider.test,
        options: {
          ...cfg.provider.test.options,
          baseURL: url,
          ...options,
        },
      },
    },
  }
}

function agent(): Agent.Info {
  return {
    name: "build",
    mode: "primary",
    options: {},
    permission: [{ permission: "*", pattern: "*", action: "allow" }],
  }
}

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const waitFor = <A>(check: Effect.Effect<A | undefined>, message: string) =>
  Effect.gen(function* () {
    const stop = Date.now() + 500
    while (Date.now() < stop) {
      const value = yield* check
      if (value !== undefined) return value
      yield* Effect.sleep("10 millis")
    }
    return yield* Effect.fail(new Error(message))
  })

const user = Effect.fn("TestSession.user")(function* (sessionID: SessionID, text: string) {
  const session = yield* Session.Service
  const msg = yield* session.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID,
    agent: "build",
    model: ref,
    time: { created: Date.now() },
  })
  yield* session.updatePart({
    id: PartID.ascending(),
    messageID: msg.id,
    sessionID,
    type: "text",
    text,
  })
  return msg
})

const assistant = Effect.fn("TestSession.assistant")(function* (
  sessionID: SessionID,
  parentID: MessageID,
  root: string,
) {
  const session = yield* Session.Service
  const msg: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    sessionID,
    mode: "build",
    agent: "build",
    path: { cwd: root, root },
    cost: 0,
    tokens: {
      total: 0,
      input: 0,
      output: 0,
      reasoning: 0,
      cache: { read: 0, write: 0 },
    },
    modelID: ref.modelID,
    providerID: ref.providerID,
    parentID,
    time: { created: Date.now() },
    finish: "end_turn",
  }
  yield* session.updateMessage(msg)
  return msg
})

const root = LayerNode.group([
  SessionProcessor.node,
  Session.node,
  SessionProjector.node,
  Provider.node,
  Database.node,
  EventV2Bridge.node,
  SessionStatus.node,
  CrossSpawnSpawner.node,
])
const replacements = [
  [SessionSummary.node, summary],
  [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
] as const
const env = LayerNode.compile(
  LayerNode.group([root, LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })]),
  replacements,
)

function postToolRecoveryLLMLayer() {
  let calls = 0
  return Layer.effect(
    LLM.Service,
    Effect.gen(function* () {
      const live = yield* LLM.Service
      return LLM.Service.of({
        stream: (input) => {
          calls += 1
          const stream = live.stream(input)
          if (calls !== 1) return stream
          return stream.pipe(
            Stream.drain,
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError(
                  "stream_incomplete: Upstream closed before response.completed after graceful tool failure",
                  {
                    transport: "websocket",
                    phase: "after_first_event",
                    autoReplaySafe: true,
                    retryable: true,
                  },
                ),
              ),
            ),
          )
        },
      })
    }),
  ).pipe(Layer.provide(LayerNode.compile(LLM.node)))
}

const postToolRecoveryEnv = LayerNode.compile(
  LayerNode.group([root, LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] })]),
  [...replacements, [LLM.node, postToolRecoveryLLMLayer()]],
)

function processorLayer(
  llmLayer: Layer.Layer<LLM.Service>,
  sessionLayer?: Layer.Layer<Session.Service>,
) {
  if (sessionLayer) {
    return LayerNode.compile(root, [...replacements, [LLM.node, llmLayer], [Session.node, sessionLayer]])
  }
  return LayerNode.compile(root, [...replacements, [LLM.node, llmLayer]])
}

function rollbackFailureSessionLayer() {
  let failed = false
  return Layer.effect(
    Session.Service,
    Effect.gen(function* () {
      const session = yield* Session.Service
      return Session.Service.of({
        ...session,
        removePart: (input: Parameters<Session.Interface["removePart"]>[0]) =>
          failed
            ? session.removePart(input)
            : Effect.sync(() => {
                failed = true
                throw new Error("rollback remove failed")
              }),
      })
    }),
  ).pipe(Layer.provide(LayerNode.compile(Session.node)))
}

const it = testEffect(env)
const postToolRecoveryIt = testEffect(postToolRecoveryEnv)
const isolatedIt = testEffect(LayerNode.compile(CrossSpawnSpawner.node))

const providerErrorLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () =>
      Stream.make(
        LLMEvent.stepStart({ index: 0 }),
        LLMEvent.toolInputStart({ id: "call-1", name: "lookup" }),
        LLMEvent.toolInputEnd({ id: "call-1", name: "lookup" }),
        LLMEvent.toolCall({ id: "call-1", name: "lookup", input: {}, providerExecuted: true }),
        LLMEvent.toolResult({
          id: "call-1",
          name: "lookup",
          result: { type: "error", value: "provider boom" },
          providerExecuted: true,
        }),
        LLMEvent.stepFinish({ index: 0, reason: "stop" }),
        LLMEvent.finish({ reason: "stop" }),
      ),
  }),
)
const providerErrorEnv = LayerNode.compile(root, [...replacements, [LLM.node, providerErrorLLM]])
const itProviderError = testEffect(providerErrorEnv)

const fragmentFailureLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () =>
      Stream.make(
        LLMEvent.stepStart({ index: 0 }),
        LLMEvent.reasoningStart({ id: "reasoning-1" }),
        LLMEvent.reasoningDelta({ id: "reasoning-1", text: "thinking" }),
        LLMEvent.textStart({ id: "text-1" }),
        LLMEvent.textDelta({ id: "text-1", text: "partial" }),
        LLMEvent.providerError({ message: "provider boom" }),
      ),
  }),
)
const fragmentFailureEnv = LayerNode.compile(root, [...replacements, [LLM.node, fragmentFailureLLM]])
const itFragmentFailure = testEffect(fragmentFailureEnv)

const boot = Effect.fn("test.boot")(function* () {
  const processors = yield* SessionProcessor.Service
  const session = yield* Session.Service
  const provider = yield* Provider.Service
  return { processors, session, provider }
})

function llmStub() {
  const queue: Array<Stream.Stream<LLMEvent, unknown>> = []
  let calls = 0

  return {
    push(...streams: Array<Stream.Stream<LLMEvent, unknown>>) {
      queue.push(...streams)
    },
    get calls() {
      return calls
    },
    layer: Layer.succeed(
      LLM.Service,
      LLM.Service.of({
        stream: () => {
          calls += 1
          return queue.shift() ?? Stream.empty
        },
      }),
    ),
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

it.live("session.processor effect tests capture llm input cleanly", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.text("hello")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "hi")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const input = {
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "hi" }],
          tools: {},
        } satisfies LLM.StreamInput

        const value = yield* handle.process(input)
        const parts = yield* MessageV2.parts(msg.id)
        const calls = yield* llm.calls

        expect(value).toBe("continue")
        expect(calls).toBe(1)
        expect(parts.some((part) => part.type === "text" && part.text === "hello")).toBe(true)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests preserve text start time", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const gate = defer<void>()
        const { processors, session, provider } = yield* boot()

        yield* llm.push(
          raw({
            head: [
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: { role: "assistant" } }],
              },
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: { content: "hello" } }],
              },
            ],
            wait: gate.promise,
            tail: [
              {
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                choices: [{ delta: {}, finish_reason: "stop" }],
              },
            ],
          }),
        )

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "hi")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "hi" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* waitFor(
          MessageV2.parts(msg.id).pipe(
            Effect.map((parts) => parts.find((part): part is SessionV1.TextPart => part.type === "text")),
            Effect.provideService(Database.Service, database),
          ),
          "timed out waiting for text part",
        )
        yield* Effect.sleep("20 millis")
        gate.resolve()

        const exit = yield* Fiber.await(run)
        const text = (yield* MessageV2.parts(msg.id)).find((part): part is SessionV1.TextPart => part.type === "text")

        expect(Exit.isSuccess(exit)).toBe(true)
        expect(text?.text).toBe("hello")
        expect(text?.time?.start).toBeDefined()
        expect(text?.time?.end).toBeDefined()
        if (!text?.time?.start || !text.time.end) return
        expect(text.time.start).toBeLessThan(text.time.end)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests stop after token overflow requests compaction", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.text("after", { usage: { input: 100, output: 0 } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "compact")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const base = yield* provider.getModel(ref.providerID, ref.modelID)
        const mdl = { ...base, limit: { context: 20, output: 10 } }
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "compact" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("compact")
        expect(parts.some((part) => part.type === "text" && part.text === "after")).toBe(true)
        expect(parts.some((part) => part.type === "step-finish")).toBe(true)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests capture reasoning from http mock", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.push(reply().reason("think").text("done").stop())

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "reason")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "reason" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)
        const reasoning = parts.find((part): part is SessionV1.ReasoningPart => part.type === "reasoning")
        const text = parts.find((part): part is SessionV1.TextPart => part.type === "text")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(1)
        expect(reasoning?.text).toBe("think")
        expect(text?.text).toBe("done")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests reset reasoning state across retries", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.push(reply().reason("one").reset(), reply().reason("two").text("done").stop())

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "reason")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "reason" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)
        const reasoning = parts.filter((part): part is SessionV1.ReasoningPart => part.type === "reasoning")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(reasoning.some((part) => part.text === "two")).toBe(true)
        expect(reasoning.some((part) => part.text === "onetwo")).toBe(false)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests do not retry unknown json errors", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(400, { error: { message: "no_kv_space" } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "json" }],
          tools: {},
        })

        expect(value).toBe("stop")
        expect(yield* llm.calls).toBe(1)
        expect(handle.message.error?.name).toBe("APIError")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests retry recognized structured json errors", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(429, { type: "error", error: { type: "too_many_requests" } })
        yield* llm.text("after")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry json" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(parts.some((part) => part.type === "text" && part.text === "after")).toBe(true)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests retry OpenAI-compatible midstream server errors", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.push(raw({ chunks: [{ error: { type: "server_error", code: "server_error", message: "xxx" } }] }))
        yield* llm.text("after")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry midstream server error")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry midstream server error" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(parts.some((part) => part.type === "text" && part.text === "after")).toBe(true)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests publish retry status updates", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        yield* llm.error(503, { error: "boom" })
        yield* llm.text("after")

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const states: number[] = []
        const off = yield* events.listen((evt) => {
          if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
          const data = evt.data as typeof SessionStatus.Event.Status.data.Type
          if (data.sessionID === chat.id && data.status.type === "retry") states.push(data.status.attempt)
          return Effect.void
        })
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry" }],
          tools: {},
        })

        yield* off

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(states).toStrictEqual([1])
      }),
    { config: (url) => providerCfg(url) },
  ),
)

isolatedIt.live("session.processor effect tests retry native stream_incomplete LLMError with visible status", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.fail(
            new LLMError({
              module: "Route",
              method: "frames",
              reason: new InvalidProviderOutputReason({
                message: "Failed to read openai-responses stream",
                route: "openai-responses",
                raw: '{"code":"stream_incomplete","message":"Upstream websocket closed before response.completed"}',
              }),
            }),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "after" }),
            LLMEvent.textEnd({ id: "text_1" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()
          const events = yield* EventV2Bridge.Service

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "native stream incomplete")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const states: number[] = []
          const off = yield* events.listen((evt) => {
            if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
            const data = evt.data as typeof SessionStatus.Event.Status.data.Type
            if (data.sessionID === chat.id && data.status.type === "retry") states.push(data.status.attempt)
            return Effect.void
          })
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "native stream incomplete" }],
            tools: {},
          })

          yield* off

          expect(value).toBe("continue")
          expect(llm.calls).toBe(2)
          expect(states).toStrictEqual([1])
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests retry exact header timeout abort without user cancellation", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.fail(new DOMException("Provider response headers timed out after 10000ms", "AbortError")),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "after" }),
            LLMEvent.textEnd({ id: "text_1" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()
          const events = yield* EventV2Bridge.Service

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "header timeout")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const states: number[] = []
          const off = yield* events.listen((evt) => {
            if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
            const data = evt.data as typeof SessionStatus.Event.Status.data.Type
            if (data.sessionID === chat.id && data.status.type === "retry") states.push(data.status.attempt)
            return Effect.void
          })
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "header timeout" }],
            tools: {},
          })

          yield* off

          expect(value).toBe("continue")
          expect(llm.calls).toBe(2)
          expect(states).toStrictEqual([1])
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests treats user-cancelled stream failure as abort without retry", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.fail(
            new ProviderError.ResponseStreamError("stream_incomplete: Upstream closed before response.completed", {
              transport: "sse",
              phase: "before_first_event",
              autoReplaySafe: true,
            }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()
          const events = yield* EventV2Bridge.Service

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "cancelled stream incomplete")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const states: number[] = []
          const off = yield* events.listen((evt) => {
            if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
            const data = evt.data as typeof SessionStatus.Event.Status.data.Type
            if (data.sessionID === chat.id && data.status.type === "retry") states.push(data.status.attempt)
            return Effect.void
          })
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
            wasCancelled: Effect.succeed(true),
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "cancelled stream incomplete" }],
            tools: {},
          })

          yield* off

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(states).toStrictEqual([])
          expect(handle.message.error?.name).toBe("MessageAbortedError")
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests rollback partial output before malformed stream retry", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new JSONParseError({
                  text: '{"choices":[{"index":0data: {"id":"abc"}',
                  cause: new SyntaxError("Expected ','"),
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "after" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry malformed stream")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry malformed stream" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("continue")
          expect(llm.calls).toBe(2)
          expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([
            "after",
          ])
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not replay malformed streams after orphan tool-input deltas", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(LLMEvent.toolInputDelta({ id: "call_delta", name: "lookup", text: '{"query":' })).pipe(
            Stream.concat(
              Stream.fail(
                new JSONParseError({
                  text: '{"type":"response.function_call_arguments.delta"',
                  cause: new SyntaxError("Unexpected end of JSON input"),
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "must not run" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "do not retry orphan tool delta")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "do not retry orphan tool delta" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          const toolPart = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(toolPart?.state.status).toBe("error")
          expect(handle.message.error).toBeDefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests rollback retryable stream_incomplete provider-error events", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
            LLMEvent.providerError({
              message: "stream_incomplete: Upstream websocket closed before response.completed",
              retryable: true,
            }),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "after" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()
          const events = yield* EventV2Bridge.Service

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "provider stream incomplete")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const states: number[] = []
          const off = yield* events.listen((evt) => {
            if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
            const data = evt.data as typeof SessionStatus.Event.Status.data.Type
            if (data.sessionID === chat.id && data.status.type === "retry") states.push(data.status.attempt)
            return Effect.void
          })
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "provider stream incomplete" }],
            tools: {},
          })

          yield* off

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("continue")
          expect(llm.calls).toBe(2)
          expect(states).toStrictEqual([1])
          expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([
            "after",
          ])
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests rollback replay-aware transient plain provider errors", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
            LLMEvent.providerError({
              message: "server_error: Upstream model unavailable",
              retryable: true,
              providerMetadata: { openai: { autoReplaySafe: false } },
            }),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "after" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry replay-aware provider error")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry replay-aware provider error" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("continue")
          expect(llm.calls).toBe(2)
          expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([
            "after",
          ])
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests surface empty response after rolling back retryable output", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
            LLMEvent.providerError({
              message: "server_error: Upstream model unavailable",
              retryable: true,
              providerMetadata: { openai: { terminalEvent: "response.failed", autoReplaySafe: false } },
            }),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry into empty response")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry into empty response" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("stop")
          expect(llm.calls).toBe(2)
          expect(parts.filter((part) => part.type === "text" || part.type === "tool")).toEqual([])
          expect(SessionV1.EmptyResponseError.isInstance(handle.message.error)).toBe(true)
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not replay arbitrary retryable provider-error events", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.providerError({
              message: "temporary provider failure",
              retryable: true,
            }),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_after" }),
            LLMEvent.textDelta({ id: "text_after", text: "after" }),
            LLMEvent.textEnd({ id: "text_after" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()
          const events = yield* EventV2Bridge.Service

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "generic retryable provider error")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const states: number[] = []
          const off = yield* events.listen((evt) => {
            if (evt.type !== SessionStatus.Event.Status.type) return Effect.void
            const data = evt.data as typeof SessionStatus.Event.Status.data.Type
            if (data.sessionID === chat.id && data.status.type === "retry") states.push(data.status.attempt)
            return Effect.void
          })
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "generic retryable provider error" }],
            tools: {},
          })

          yield* off

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(states).toStrictEqual([])
          expect(handle.message.error).toBeDefined()
          expect(handle.message.error && "message" in handle.message.error.data ? handle.message.error.data.message : "").toContain(
            "temporary provider failure",
          )
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests compact on classified provider-error context overflow", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.providerError({
              message: "context_length_exceeded: prompt too long",
              classification: "context-overflow",
            }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "overflow provider error")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "overflow provider error" }],
            tools: {},
          })

          expect(value).toBe("compact")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests compact on native HTTP context overflow LLMError", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.fail(
            new LLMError({
              module: "RequestExecutor",
              method: "execute",
              reason: new InvalidRequestReason({
                message: "context_length_exceeded: prompt too long",
                classification: "context-overflow",
              }),
            }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "native overflow")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "native overflow" }],
            tools: {},
          })

          expect(value).toBe("compact")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

it.live("session.processor effect tests rollback assistant-only partial output before retry", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.push(reply().text("partial").hang().item(), reply().text("after").stop().item())

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry partial output")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry partial output" }],
          tools: {},
        })

        const parts = yield* MessageV2.parts(msg.id)

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(2)
        expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
        expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([
          "after",
        ])
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfgWithOptions(url, { chunkTimeout: 50 }) },
  ),
)

isolatedIt.live("session.processor effect tests rollback retryable websocket response.done after partial output", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError(
                  "stream_incomplete: Upstream websocket closed before response.completed",
                  {
                    transport: "websocket",
                    phase: "after_first_event",
                    autoReplaySafe: false,
                    retryable: true,
                    terminalEvent: "response.done",
                  },
                ),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "after" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "do not retry response.done after output")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "do not retry response.done after output" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          expect(value).toBe("continue")
          expect(llm.calls).toBe(2)
          expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([
            "after",
          ])
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests rollback a completed assistant-only step before transient terminal retry", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.reasoningStart({ id: "reasoning_1" }),
            LLMEvent.reasoningDelta({ id: "reasoning_1", text: "discarded reasoning" }),
            LLMEvent.reasoningEnd({ id: "reasoning_1" }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "discarded answer" }),
            LLMEvent.textEnd({ id: "text_1" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop", usage: { inputTokens: 2, outputTokens: 3, totalTokens: 5 } }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("server_error: response failed after the completed step", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                  retryable: true,
                  terminalEvent: "response.failed",
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "after" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop", usage: { inputTokens: 5, outputTokens: 7, totalTokens: 12 } }),
            LLMEvent.finish({ reason: "stop", usage: { inputTokens: 5, outputTokens: 7, totalTokens: 12 } }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry completed assistant-only step")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry completed assistant-only step" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("continue")
          expect(llm.calls).toBe(2)
          expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(parts.filter((part) => part.type === "step-finish")).toHaveLength(1)
          expect(parts.filter((part): part is SessionV1.ReasoningPart => part.type === "reasoning")).toEqual([])
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([
            "after",
          ])
          expect(handle.message.tokens).toEqual({
            input: 5,
            output: 7,
            total: 12,
            reasoning: 0,
            cache: { read: 0, write: 0 },
          })
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not replay a completed text plugin effect", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const marker = path.join(dir, "text-complete-effects.txt")
        const pluginFile = path.join(dir, "text-complete-plugin.ts")
        yield* Effect.promise(() => Bun.write(marker, ""))
        yield* Effect.promise(() =>
          Bun.write(
            pluginFile,
            [
              "export default async () => ({",
              '  "experimental.text.complete": async () => {',
              `    await Bun.write(${JSON.stringify(marker)}, (await Bun.file(${JSON.stringify(marker)}).text()) + "complete\\n")`,
              "  },",
              "})",
              "",
            ].join("\n"),
          ),
        )
        yield* Effect.promise(() =>
          Bun.write(
            path.join(dir, "opencode.json"),
            JSON.stringify({
              $schema: "https://opencode.ai/config.json",
              ...cfg,
              plugin: [pathToFileURL(pluginFile).href],
            }),
          ),
        )

        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "completed answer" }),
            LLMEvent.textEnd({ id: "text_1" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("server_error: failed after the completed plugin effect", {
                  transport: "websocket",
                  phase: "before_first_event",
                  autoReplaySafe: true,
                  retryable: true,
                  terminalEvent: "response.failed",
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "must not run" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "do not replay completed plugin effects")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "do not replay completed plugin effects" }],
            tools: {},
          })

          const effects = yield* Effect.promise(() => Bun.file(marker).text())

          expect(effects).toBe("complete\n")
          expect(llm.calls).toBe(1)
          expect(value).toBe("stop")
          expect(handle.message.error).toBeDefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not retry a completed step with a deterministic empty-response error", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.stepFinish({ index: 0, reason: "stop", usage: { inputTokens: 2, outputTokens: 0, totalTokens: 2 } }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("server_error: response failed after the empty step", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                  retryable: true,
                  terminalEvent: "response.failed",
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "must not run" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "do not retry deterministic empty response")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "do not retry deterministic empty response" }],
            tools: {},
          })

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(SessionV1.EmptyResponseError.isInstance(handle.message.error)).toBe(true)
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

for (const reason of ["content-filter", "error"] as const) {
  isolatedIt.live(`session.processor effect tests do not retry completed deterministic finish reason ${reason}`, () =>
    provideTmpdirInstance(
      (dir) =>
        Effect.gen(function* () {
          const llm = llmStub()
          llm.push(
            Stream.make(
              LLMEvent.stepStart({ index: 0 }),
              LLMEvent.textStart({ id: "text_1" }),
              LLMEvent.textDelta({ id: "text_1", text: "deterministic output" }),
              LLMEvent.textEnd({ id: "text_1" }),
              LLMEvent.stepFinish({ index: 0, reason }),
            ).pipe(
              Stream.concat(
                Stream.fail(
                  new ProviderError.ResponseStreamError(`server_error: response failed after ${reason}`, {
                    transport: "websocket",
                    phase: "after_first_event",
                    autoReplaySafe: false,
                    retryable: true,
                    terminalEvent: "response.failed",
                  }),
                ),
              ),
            ),
            Stream.make(
              LLMEvent.stepStart({ index: 0 }),
              LLMEvent.textStart({ id: "text_2" }),
              LLMEvent.textDelta({ id: "text_2", text: "must not run" }),
              LLMEvent.textEnd({ id: "text_2" }),
              LLMEvent.stepFinish({ index: 0, reason: "stop" }),
              LLMEvent.finish({ reason: "stop" }),
            ),
          )

          const effect = Effect.gen(function* () {
            const { processors, session, provider } = yield* boot()

            const chat = yield* session.create({})
            const parent = yield* user(chat.id, `do not retry ${reason}`)
            const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
            const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
            const handle = yield* processors.create({
              assistantMessage: msg,
              sessionID: chat.id,
              model: mdl,
            })

            const value = yield* handle.process({
              user: {
                id: parent.id,
                sessionID: chat.id,
                role: "user",
                time: parent.time,
                agent: parent.agent,
                model: { providerID: ref.providerID, modelID: ref.modelID },
              } satisfies SessionV1.User,
              sessionID: chat.id,
              model: mdl,
              agent: agent(),
              system: [],
              messages: [{ role: "user", content: `do not retry ${reason}` }],
              tools: {},
            })

            expect(value).toBe("stop")
            expect(llm.calls).toBe(1)
            expect(handle.message.finish).toBe(reason)
          })

          yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
        }),
      { config: cfg },
    ),
  )
}

isolatedIt.live("session.processor effect tests keep content-filter terminal after a settled tool", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls" }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "filtered answer" }),
            LLMEvent.textEnd({ id: "text_1" }),
            LLMEvent.stepFinish({ index: 1, reason: "content-filter" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("server_error: response failed after content filter", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                  retryable: true,
                  terminalEvent: "response.failed",
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "keep content filter terminal after tool")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "keep content filter terminal after tool" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(handle.message.finish).toBe("content-filter")
          expect(parts.filter((part) => part.type === "tool")).toHaveLength(1)
          expect(parts.some((part) => part.type === "text" && part.text === "filtered answer")).toBe(true)
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests keep error finish terminal after a settled tool", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls" }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "error answer" }),
            LLMEvent.textEnd({ id: "text_1" }),
            LLMEvent.stepFinish({ index: 1, reason: "error" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("stream_incomplete: response.done failed after error finish", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                  retryable: true,
                  terminalEvent: "response.done",
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "keep error finish terminal after tool")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "keep error finish terminal after tool" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(handle.message.finish).toBe("error")
          expect(parts.filter((part) => part.type === "tool")).toHaveLength(1)
          expect(parts.some((part) => part.type === "text" && part.text === "error answer")).toBe(true)
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests let a filesystem patch override provider replay-safe metadata", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "answer with external effect" }),
            LLMEvent.textEnd({ id: "text_1" }),
          ).pipe(
            Stream.concat(
              Stream.fromEffect(
                Effect.promise(async () => {
                  await Bun.write(path.join(dir, "side-effect.txt"), "changed")
                  return LLMEvent.stepFinish({
                    index: 0,
                    reason: "stop",
                    usage: { inputTokens: 2, outputTokens: 3, totalTokens: 5 },
                  })
                }),
              ),
            ),
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("server_error: response failed after the patched step", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: true,
                  retryable: true,
                  terminalEvent: "response.failed",
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "must not run" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "do not retry completed step with patch")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "do not retry completed step with patch" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(parts.some((part) => part.type === "patch")).toBe(true)
          expect(parts.some((part) => part.type === "text" && part.text === "answer with external effect")).toBe(true)
          expect(handle.message.error).toBeDefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg, git: true },
  ),
)

isolatedIt.live("session.processor effect tests preserve permanent response.done provider-error metadata", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.providerError({
              message: "Rate limit reached",
              retryable: false,
              providerMetadata: { openai: { terminalEvent: "response.done", autoReplaySafe: true } },
            }),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "after" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "do not retry response.done provider error")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "do not retry response.done provider error" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(parts.filter((part) => part.type === "text" || part.type === "tool")).toEqual([])
          expect(SessionV1.APIError.isInstance(handle.message.error)).toBe(true)
          if (SessionV1.APIError.isInstance(handle.message.error)) {
            expect(handle.message.error.data.message).toBe("Rate limit reached")
            expect(handle.message.error.data.isRetryable).toBe(false)
            expect(handle.message.error.data.metadata?.terminalEvent).toBe("response.done")
            expect(handle.message.error.data.metadata?.autoReplaySafe).toBe("true")
            expect(handle.message.error.data.metadata?.retryable).toBe("false")
          }
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests resume from rebuilt history after a post-tool stream failure", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("Upstream websocket closed before response.completed", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry after tool step")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const input = {
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry after tool step" }],
            tools: {},
          } satisfies LLM.StreamInput

          const value = yield* handle.process(input)

          expect(value).toBe("resume")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeUndefined()
          expect(handle.message.finish).toBe("tool-calls")

          const resumed = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const history = yield* MessageV2.filterCompactedEffect(chat.id)
          const resumedHandle = yield* processors.create({
            assistantMessage: resumed,
            sessionID: chat.id,
            model: mdl,
          })
          const resumedValue = yield* resumedHandle.process({
            ...input,
            messages: yield* MessageV2.toModelMessagesEffect(history, mdl),
          })

          const firstParts = yield* MessageV2.parts(msg.id)
          const resumedParts = yield* MessageV2.parts(resumed.id)
          const toolParts = (yield* MessageV2.filterCompactedEffect(chat.id))
            .flatMap((entry) => entry.parts)
            .filter((part): part is SessionV1.ToolPart => part.type === "tool")

          expect(resumedValue).toBe("continue")
          expect(llm.calls).toBe(2)
          expect(firstParts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(firstParts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([])
          expect(resumedParts.filter((part) => part.type === "tool")).toHaveLength(0)
          expect(toolParts).toHaveLength(1)
          expect(toolParts[0]?.state.status).toBe("completed")
          expect(resumedHandle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not resume unsafe response.failed after model output", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("OpenAI response failed (server_error): failed", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                  terminalEvent: "response.failed",
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "do not retry response.failed after output")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "do not retry response.failed after output" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          const text = parts.find((part): part is SessionV1.TextPart => part.type === "text")

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(text?.text).toBe("partial")
          expect(SessionV1.APIError.isInstance(handle.message.error)).toBe(true)
          if (SessionV1.APIError.isInstance(handle.message.error)) {
            expect(handle.message.error.data.metadata?.terminalEvent).toBe("response.failed")
            expect(handle.message.error.data.metadata?.autoReplaySafe).toBe("false")
          }
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests resume after a completed tool result and transient response.failed", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("server_error: response failed after the completed tool", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                  retryable: true,
                  terminalEvent: "response.failed",
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry immediately after tool result")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry immediately after tool result" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          const toolPart = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

          expect(value).toBe("resume")
          expect(llm.calls).toBe(1)
          expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(parts.filter((part) => part.type === "tool")).toHaveLength(1)
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([])
          expect(toolPart?.state.status).toBe("completed")
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not retry plain-text rate limit errors after a completed tool boundary", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
          ).pipe(Stream.concat(Stream.fail(new Error("Too many requests")))),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry plain text after tool result")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry plain text after tool result" }],
            tools: {},
          })

          expect(value).toBe("resume")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests resume immediately after a completed tool result on retryable HTTP failure", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new APICallError({
                  message: "boom",
                  url: "https://example.com/v1/chat/completions",
                  requestBodyValues: {},
                  statusCode: 500,
                  responseHeaders: { "content-type": "application/json" },
                  responseBody: '{"error":"boom"}',
                  isRetryable: true,
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry http after tool result")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry http after tool result" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          const toolPart = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

          expect(value).toBe("resume")
          expect(llm.calls).toBe(1)
          expect(toolPart?.state.status).toBe("completed")
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([])
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests resume after post-finish tool results", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("Upstream websocket closed before response.completed", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "after" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 1, reason: "stop", usage: { inputTokens: 5, outputTokens: 8, totalTokens: 13 } }),
            LLMEvent.finish({ reason: "stop", usage: { inputTokens: 5, outputTokens: 8, totalTokens: 13 } }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry after post-finish tool result")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry after post-finish tool result" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)

          expect(value).toBe("resume")
          expect(llm.calls).toBe(1)
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([])
          expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(parts.filter((part) => part.type === "step-finish")).toHaveLength(1)
          expect(handle.message.error).toBeUndefined()
          expect(handle.message.finish).toBe("tool-calls")
          expect(handle.message.tokens).toEqual({
            input: 1,
            output: 1,
            total: 2,
            reasoning: 0,
            cache: { read: 0, write: 0 },
          })
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests stop when committed-boundary rollback fails", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
          ).pipe(Stream.concat(Stream.fail(new Error("Too many requests")))),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "rollback failure after tool")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "rollback failure after tool" }],
            tools: {},
          })

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeDefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer, rollbackFailureSessionLayer())))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests stop when model-only JSON rollback fails", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new JSONParseError({
                  text: '{"type":"response.output_text.delta"',
                  cause: new SyntaxError("Unexpected end of JSON input"),
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "must not run" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "rollback failure for malformed model output")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "rollback failure for malformed model output" }],
            tools: {},
          })

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeDefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer, rollbackFailureSessionLayer())))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests stop when model-only native LLM rollback fails", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_1" }),
            LLMEvent.textDelta({ id: "text_1", text: "partial" }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new LLMError({
                  module: "Route",
                  method: "frames",
                  reason: new InvalidProviderOutputReason({
                    message: "Failed to read openai-responses stream",
                    route: "openai-responses",
                    raw: '{"code":"stream_incomplete","message":"Upstream websocket closed before response.completed"}',
                  }),
                }),
              ),
            ),
          ),
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.textStart({ id: "text_2" }),
            LLMEvent.textDelta({ id: "text_2", text: "must not run" }),
            LLMEvent.textEnd({ id: "text_2" }),
            LLMEvent.stepFinish({ index: 0, reason: "stop" }),
            LLMEvent.finish({ reason: "stop" }),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "rollback failure for native stream error")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "rollback failure for native stream error" }],
            tools: {},
          })

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeDefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer, rollbackFailureSessionLayer())))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests resume after a durable tool error and transient response.done", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolError({
              id: "call_1",
              name: "lookup",
              message: "lookup failed",
              error: new Error("lookup failed"),
            }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("stream_incomplete: response.done failed after the tool error", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                  retryable: true,
                  terminalEvent: "response.done",
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry after tool error")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry after tool error" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          const toolPart = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

          expect(value).toBe("resume")
          expect(llm.calls).toBe(1)
          expect(parts.filter((part) => part.type === "step-start")).toHaveLength(1)
          expect(parts.filter((part) => part.type === "tool")).toHaveLength(1)
          expect(parts.filter((part): part is SessionV1.TextPart => part.type === "text").map((part) => part.text)).toEqual([])
          expect(toolPart?.state.status).toBe("error")
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not resume when a new tool step is in flight after a committed boundary", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.toolCall({ id: "call_2", name: "lookup", input: { query: "forecast" } }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("Upstream websocket closed before response.completed", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry during new tool step")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry during new tool step" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          const completed = parts.find(
            (part): part is SessionV1.ToolPart =>
              part.type === "tool" && part.callID === "call_1" && part.state.status === "completed",
          )
          const interrupted = parts.find(
            (part): part is SessionV1.ToolPart =>
              part.type === "tool" && part.callID === "call_2" && part.state.status === "error",
          )

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(completed?.state.status).toBe("completed")
          expect(interrupted?.state.status).toBe("error")
          if (interrupted?.state.status === "error") {
            expect(interrupted.state.error).toBe("Tool execution aborted")
            expect(interrupted.state.metadata?.interrupted).toBe(true)
          }
          expect(handle.message.error).toBeDefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not retry API failures during a new tool step after a committed boundary", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.toolCall({ id: "call_2", name: "lookup", input: { query: "forecast" } }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new APICallError({
                  message: "boom",
                  url: "https://example.com/v1/chat/completions",
                  requestBodyValues: {},
                  statusCode: 500,
                  responseHeaders: { "content-type": "application/json" },
                  responseBody: '{"error":"boom"}',
                  isRetryable: true,
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry api during new tool step")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry api during new tool step" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          const completed = parts.find(
            (part): part is SessionV1.ToolPart =>
              part.type === "tool" && part.callID === "call_1" && part.state.status === "completed",
          )
          const interrupted = parts.find(
            (part): part is SessionV1.ToolPart =>
              part.type === "tool" && part.callID === "call_2" && part.state.status === "error",
          )

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(completed?.state.status).toBe("completed")
          expect(interrupted?.state.status).toBe("error")
          if (interrupted?.state.status === "error") {
            expect(interrupted.state.error).toBe("Tool execution aborted")
            expect(interrupted.state.metadata?.interrupted).toBe(true)
          }
          expect(handle.message.error?.name).toBe("APIError")
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not retry plain-text rate limit errors during a new tool step", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.toolCall({ id: "call_2", name: "lookup", input: { query: "forecast" } }),
          ).pipe(Stream.concat(Stream.fail(new Error("Too many requests")))),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry plain text during new tool step")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry plain text during new tool step" }],
            tools: {},
          })

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeDefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests stop after denied tool boundary even when a retryable failure follows", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.toolCall({ id: "call_2", name: "lookup", input: { query: "forecast" } }),
            LLMEvent.toolError({
              id: "call_2",
              name: "lookup",
              message: "permission denied",
              error: new PermissionV1.RejectedError(),
            }),
          ).pipe(
            Stream.concat(
              Stream.fail(
                new ProviderError.ResponseStreamError("Upstream websocket closed before response.completed", {
                  transport: "websocket",
                  phase: "after_first_event",
                  autoReplaySafe: false,
                }),
              ),
            ),
          ),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry after denied tool")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry after denied tool" }],
            tools: {},
          })

          const parts = yield* MessageV2.parts(msg.id)
          const denied = parts.find(
            (part): part is SessionV1.ToolPart =>
              part.type === "tool" && part.callID === "call_2" && part.state.status === "error",
          )

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(denied?.state.status).toBe("error")
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

isolatedIt.live("session.processor effect tests do not retry plain-text rate limit errors after a denied tool", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const llm = llmStub()
        llm.push(
          Stream.make(
            LLMEvent.stepStart({ index: 0 }),
            LLMEvent.toolCall({ id: "call_1", name: "lookup", input: { query: "weather" } }),
            LLMEvent.toolResult({
              id: "call_1",
              name: "lookup",
              result: { type: "json", value: { title: "Weather lookup", output: "result:weather", metadata: {} } },
            }),
            LLMEvent.stepFinish({ index: 0, reason: "tool-calls", usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }),
            LLMEvent.stepStart({ index: 1 }),
            LLMEvent.toolCall({ id: "call_2", name: "lookup", input: { query: "forecast" } }),
            LLMEvent.toolError({
              id: "call_2",
              name: "lookup",
              message: "permission denied",
              error: new PermissionV1.RejectedError(),
            }),
          ).pipe(Stream.concat(Stream.fail(new Error("Too many requests")))),
        )

        const effect = Effect.gen(function* () {
          const { processors, session, provider } = yield* boot()

          const chat = yield* session.create({})
          const parent = yield* user(chat.id, "retry plain text after denied tool")
          const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
          const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
          const handle = yield* processors.create({
            assistantMessage: msg,
            sessionID: chat.id,
            model: mdl,
          })

          const value = yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "retry plain text after denied tool" }],
            tools: {},
          })

          expect(value).toBe("stop")
          expect(llm.calls).toBe(1)
          expect(handle.message.error).toBeUndefined()
        })

        yield* effect.pipe(Effect.provide(processorLayer(llm.layer)))
      }),
    { config: cfg },
  ),
)

it.live("session.processor effect tests do not retry partial output after tool activity in the current step", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.push(reply().text("partial").pendingTool("lookup", { query: "weather" }).hang().item())

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "retry tool partial output")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "retry tool partial output" }],
          tools: {},
        })

        expect(value).toBe("stop")
        expect(yield* llm.calls).toBe(1)
        expect(handle.message.error).toBeDefined()
      }),
    { config: (url) => providerCfgWithOptions(url, { chunkTimeout: 50 }) },
  ),
)

it.live("session.processor effect tests compact on structured context overflow", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.error(400, { type: "error", error: { code: "context_length_exceeded" } })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "compact json")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "compact json" }],
          tools: {},
        })

        expect(value).toBe("compact")
        expect(yield* llm.calls).toBe(1)
        expect(handle.message.error).toBeUndefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests complete AI SDK tool calls when native flag is off", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()

        yield* llm.tool("lookup", { query: "weather" })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "tool")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "tool" }],
          tools: {
            lookup: tool({
              description: "Look up information",
              inputSchema: z.object({ query: z.string() }),
              execute: async (input) => ({
                title: "Weather lookup",
                output: `result:${input.query}`,
                metadata: { source: "test" },
              }),
            }),
          },
        })

        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

        expect(value).toBe("continue")
        expect(yield* llm.calls).toBe(1)
        expect(call?.callID).toBe("call_1")
        expect(call?.tool).toBe("lookup")
        expect(call?.state.status).toBe("completed")
        if (call?.state.status !== "completed") return
        expect(call.state.input).toEqual({ query: "weather" })
        expect(call.state.output).toBe("result:weather")
        expect(call.state.title).toBe("Weather lookup")
        expect(call.state.metadata).toEqual({ source: "test" })
        expect(call.state.time.start).toBeDefined()
        expect(call.state.time.end).toBeDefined()
      }),
    { config: (url) => providerCfg(url) },
  ),
)

postToolRecoveryIt.live("session.processor effect tests retry a graceful tool failure hidden by stream incomplete", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        let executions = 0
        const receivers: unknown[] = []

        yield* llm.push(
          reply().tool("lookup", { query: "weather" }),
          reply().tool("lookup", { query: "weather" }),
        )

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "recover after one real tool execution")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const lookup = tool({
          description: "Look up information",
          inputSchema: z.object({ query: z.string() }),
          async execute(input) {
            executions += 1
            receivers.push(this)
            if (executions === 1) throw new Error("graceful lookup failure")
            return {
              title: "Weather lookup",
              output: `result:${input.query}`,
              metadata: { source: "test" },
            }
          },
        })
        Object.freeze(lookup)
        const executeDescriptor = Object.getOwnPropertyDescriptor(lookup, "execute")
        const prototype = Object.getPrototypeOf(lookup)
        const tools = { lookup }
        const input = {
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "recover after one real tool execution" }],
          tools,
        } satisfies LLM.StreamInput
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const value = yield* handle.process(input)
        const toolParts = (yield* MessageV2.parts(msg.id)).filter(
          (part): part is SessionV1.ToolPart => part.type === "tool",
        )

        expect(executions).toBe(2)
        expect(receivers).toEqual([lookup, lookup])
        expect(yield* llm.calls).toBe(2)
        expect(value).toBe("continue")
        expect(toolParts).toHaveLength(1)
        expect(toolParts[0]?.state.status).toBe("completed")
        expect(handle.message.error).toBeUndefined()
        expect(Object.getOwnPropertyDescriptor(lookup, "execute")).toEqual(executeDescriptor)
        expect(Object.getPrototypeOf(lookup)).toBe(prototype)
        expect(Object.isFrozen(lookup)).toBe(true)
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests mark pending tools as aborted on cleanup", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const database = yield* Database.Service
        const { processors, session, provider } = yield* boot()

        yield* llm.toolHang("bash", { cmd: "pwd" })

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "tool abort")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "tool abort" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* waitFor(
          MessageV2.parts(msg.id).pipe(
            Effect.map((parts) => parts.find((part): part is SessionV1.ToolPart => part.type === "tool")),
            Effect.provideService(Database.Service, database),
          ),
          "timed out waiting for tool part",
        )
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")

        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.hasInterruptsOnly(exit.cause)).toBe(true)
        }
        expect(yield* llm.calls).toBe(1)
        expect(call?.state.status).toBe("error")
        if (call?.state.status === "error") {
          expect(call.state.error).toBe("Tool execution aborted")
          expect(call.state.metadata?.interrupted).toBe(true)
          expect(call.state.time.end).toBeDefined()
        }
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests record aborted errors and idle state", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const seen = defer<void>()
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service
        const sts = yield* SessionStatus.Service

        yield* llm.hang

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "abort")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const errs: string[] = []
        const off = yield* events.listen((evt) => {
          if (evt.type !== Session.Event.Error.type) return Effect.void
          const data = evt.data as typeof Session.Event.Error.data.Type
          if (data.sessionID !== chat.id || !data.error) return Effect.void
          errs.push(data.error.name)
          seen.resolve()
          return Effect.void
        })
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "abort" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        yield* Effect.promise(() => seen.promise)
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })
        const state = yield* sts.get(chat.id)
        yield* off

        expect(Exit.isFailure(exit)).toBe(true)
        if (Exit.isFailure(exit)) {
          expect(Cause.hasInterruptsOnly(exit.cause)).toBe(true)
        }
        expect(handle.message.error?.name).toBe("MessageAbortedError")
        expect(stored.info.role).toBe("assistant")
        if (stored.info.role === "assistant") {
          expect(stored.info.error?.name).toBe("MessageAbortedError")
        }
        expect(state).toMatchObject({ type: "idle" })
        expect(errs).toContain("MessageAbortedError")
      }),
    { config: (url) => providerCfg(url) },
  ),
)

it.live("session.processor effect tests mark interruptions aborted without manual abort", () =>
  provideTmpdirServer(
    ({ dir, llm }) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const sts = yield* SessionStatus.Service

        yield* llm.hang

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "interrupt")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const handle = yield* processors.create({
          assistantMessage: msg,
          sessionID: chat.id,
          model: mdl,
        })

        const run = yield* handle
          .process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "interrupt" }],
            tools: {},
          })
          .pipe(Effect.forkChild)

        yield* llm.wait(1)
        yield* Fiber.interrupt(run)

        const exit = yield* Fiber.await(run)
        const stored = yield* MessageV2.get({ sessionID: chat.id, messageID: msg.id })
        const state = yield* sts.get(chat.id)

        expect(Exit.isFailure(exit)).toBe(true)
        expect(handle.message.error?.name).toBe("MessageAbortedError")
        expect(stored.info.role).toBe("assistant")
        if (stored.info.role === "assistant") {
          expect(stored.info.error?.name).toBe("MessageAbortedError")
        }
        expect(state).toMatchObject({ type: "idle" })
      }),
    { config: (url) => providerCfg(url) },
  ),
)

itProviderError.live("session.processor effect tests fail provider-executed error results", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "provider tool error")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const seen: string[] = []
        const off = yield* events.listen((event) => {
          seen.push(event.type)
          return Effect.void
        })
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model: mdl })

        yield* handle.process({
          user: {
            id: parent.id,
            sessionID: chat.id,
            role: "user",
            time: parent.time,
            agent: parent.agent,
            model: { providerID: ref.providerID, modelID: ref.modelID },
          } satisfies SessionV1.User,
          sessionID: chat.id,
          model: mdl,
          agent: agent(),
          system: [],
          messages: [{ role: "user", content: "provider tool error" }],
          tools: {},
        })
        yield* off

        const parts = yield* MessageV2.parts(msg.id)
        const call = parts.find((part): part is SessionV1.ToolPart => part.type === "tool")
        expect(call?.state.status).toBe("error")
        if (call?.state.status === "error") expect(call.state.error).toBe("provider boom")
        expect(seen).toContain(MessageV2.Event.PartUpdated.type)
        expect(seen).toContain(MessageV2.Event.Updated.type)
        expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
      }),
    { config: cfg },
  ),
)

itFragmentFailure.live("session.processor effect tests retain partial legacy parts without v2 events", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        const { processors, session, provider } = yield* boot()
        const events = yield* EventV2Bridge.Service

        const chat = yield* session.create({})
        const parent = yield* user(chat.id, "provider failure")
        const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
        const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
        const seen: string[] = []
        const off = yield* events.listen((event) => {
          seen.push(event.type)
          return Effect.void
        })
        const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model: mdl })

        expect(
          yield* handle.process({
            user: {
              id: parent.id,
              sessionID: chat.id,
              role: "user",
              time: parent.time,
              agent: parent.agent,
              model: { providerID: ref.providerID, modelID: ref.modelID },
            } satisfies SessionV1.User,
            sessionID: chat.id,
            model: mdl,
            agent: agent(),
            system: [],
            messages: [{ role: "user", content: "provider failure" }],
            tools: {},
          }),
        ).toBe("stop")
        yield* off

        const parts = yield* MessageV2.parts(msg.id)
        expect(parts).toEqual(
          expect.arrayContaining([
            expect.objectContaining({ type: "text", text: "partial" }),
            expect.objectContaining({ type: "reasoning", text: "thinking" }),
          ]),
        )
        expect(seen).toContain(MessageV2.Event.PartUpdated.type)
        expect(seen).toContain(Session.Event.Error.type)
        expect(seen.filter((type) => type.startsWith("session.next."))).toEqual([])
      }),
    { config: cfg },
  ),
)
