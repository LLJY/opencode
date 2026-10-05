import { describe, expect, spyOn } from "bun:test"
import { LanguageModel } from "@opencode/ai"
import { OpenAIChat } from "@opencode/ai/protocols/openai-chat"
import { TestLLM } from "@opencode/ai/testing"
import { Agent } from "@opencode/core/agent"
import { Bus } from "@opencode/core/bus"
import { Config } from "@opencode/core/config"
import { Database } from "@opencode/core/database/database"
import { AppNodeBuilder } from "@opencode/core/effect/app-node-builder"
import { LayerNodePlatform } from "@opencode/core/effect/app-node-platform"
import { EventTable } from "@opencode/core/event/sql"
import { Image } from "@opencode/core/image"
import { InstructionDiscovery } from "@opencode/core/instruction-discovery"
import { Instructions } from "@opencode/core/instructions/index"
import { InstructionBuiltIns } from "@opencode/core/instructions/builtins"
import { Location } from "@opencode/core/location"
import { McpInstructions } from "@opencode/core/mcp/instructions"
import { Model } from "@opencode/core/model"
import { Permission } from "@opencode/core/permission"
import { Plugin } from "@opencode/core/plugin"
import { PluginHooks } from "@opencode/core/plugin/hooks"
import { PluginSupervisor } from "@opencode/core/plugin/supervisor"
import { Project } from "@opencode/core/project"
import { ProjectTable } from "@opencode/core/project/sql"
import { ReferenceInstructions } from "@opencode/core/reference/instructions"
import { AbsolutePath } from "@opencode/core/schema"
import { SessionInbox } from "@opencode/core/session/inbox"
import { SessionMessage } from "@opencode/core/session/message"
import { SessionModelTransport } from "@opencode/core/session/model-transport"
import { SessionProjector } from "@opencode/core/session/projector"
import { SessionRunner } from "@opencode/core/session/runner/index"
import { SessionRunnerLLM } from "@opencode/core/session/runner/llm"
import { SessionRunnerModel } from "@opencode/core/session/runner/model"
import { SessionSchema } from "@opencode/core/session/schema"
import { SessionTable } from "@opencode/core/session/sql"
import { SessionStore } from "@opencode/core/session/store"
import { SkillInstructions } from "@opencode/core/skill/instructions"
import { Snapshot } from "@opencode/core/snapshot"
import { Tool } from "@opencode/core/tool"
import { QuestionTool } from "@opencode/core/tool/plugin/question"
import type { Info } from "@opencode/schema/tool"
import { LayerNode } from "@opencode/util/effect/layer-node"
import { asc } from "drizzle-orm"
import { Cause, Deferred, Effect, Exit, Fiber, Layer, Schema } from "effect"
import { testEffect } from "./lib/effect"
import { permissionLayer } from "./lib/permission"
import { Expected } from "./lib/session-message"

const registryLayer = AppNodeBuilder.build(LayerNode.group([Tool.node, PluginHooks.node]), [
  Image.node.replace(Layer.mock(Image.Service, {})),
])
const it = testEffect(registryLayer)
const sessionID = SessionSchema.ID.make("ses_codemode_safety")
const identity = {
  sessionID,
  agent: Agent.ID.make("build"),
  messageID: SessionMessage.ID.make("msg_codemode_safety"),
}
const execute = (snapshot: Tool.Snapshot, code: string) =>
  snapshot.execute({
    ...identity,
    call: { type: "tool-call", id: "call-execute", name: "execute", input: { code } },
  })
const leaf = (run: Info["execute"]): Info => ({
  name: "leaf",
  description: "Perform an external leaf effect",
  input: Schema.Struct({}),
  output: Schema.String,
  execute: run,
})
const declined = new Permission.DeclinedError()
const cancelled = new QuestionTool.CancelledError()
const blocked = new Tool.Error({
  message: "Permission blocked",
  error: new Permission.BlockedError({ permission: "edit", resources: ["*"], rules: [] }),
})
const stops = [
  {
    name: "invalid declared output",
    result: () => Effect.succeed({ output: 42 }),
    effects: 1,
    message: "invalid value",
  },
  { name: "missing declared output", result: () => Effect.succeed({}), effects: 1, message: "declared output" },
  {
    name: "unexpected host defect",
    result: () => Effect.die("native host defect"),
    effects: 1,
    message: "native host defect",
  },
  { name: "permission decline", result: () => Effect.die(declined), effects: 0, decision: declined },
  { name: "question cancellation", result: () => Effect.die(cancelled), effects: 0, decision: cancelled },
  { name: "wrapped permission block", result: () => Effect.fail(blocked), effects: 0, decision: blocked },
  {
    name: "non-JSON declared output",
    schema: Schema.Unknown,
    result: () => Effect.succeed({ output: 1n }),
    effects: 1,
    message: "BigInt",
  },
  {
    name: "cyclic declared output",
    schema: Schema.Unknown,
    result: () => {
      const output: Record<string, unknown> = {}
      output.self = output
      return Effect.succeed({ output })
    },
    effects: 1,
    message: "cyclic",
  },
]

const register = (tools: Tool.Interface, stop: (typeof stops)[number], effects: string[]) =>
  tools.transform((editor) => {
    editor.add({
      ...leaf(() =>
        Effect.sync(() => {
          if (stop.effects) effects.push("leaf")
        }).pipe(Effect.andThen(stop.result())),
      ),
      output: "schema" in stop ? stop.schema : Schema.String,
    })
    editor.add({
      ...leaf(() =>
        Effect.sync(() => {
          effects.push("later")
          return { output: "later" }
        }),
      ),
      name: "later",
    })
  })

const retryProgram = `
  try { await tools.leaf({}); } catch (error) {
    try { await tools.leaf({}); } catch (again) {}
  }
  await tools.later({});
  return "unsafe success";
`

describe("default Code Mode native stop boundary", () => {
  for (const stop of stops) {
    it.effect(`does not let JavaScript catch and repeat ${stop.name}`, () =>
      Effect.gen(function* () {
        const tools = yield* Tool.Service
        const effects: string[] = []
        yield* register(tools, stop, effects)
        const snapshot = yield* tools.snapshot()
        expect(snapshot.definitions.map((tool) => tool.name)).toEqual(["execute"])
        const exit = yield* execute(snapshot, retryProgram).pipe(Effect.exit)
        expect({ effects, exit: exit._tag }).toEqual({ effects: stop.effects ? ["leaf"] : [], exit: "Failure" })
        if (Exit.isFailure(exit)) {
          if (stop.message) expect(Cause.pretty(exit.cause)).toContain(stop.message)
          if (stop.decision)
            expect(
              exit.cause.reasons.some((reason) =>
                Cause.isDieReason(reason)
                  ? reason.defect === stop.decision
                  : Cause.isFailReason(reason) && reason.error === stop.decision,
              ),
            ).toBe(true)
        }
      }),
    )
  }

  it.effect("cannot recover a host defect using Promise.allSettled or finally", () =>
    Effect.gen(function* () {
      const tools = yield* Tool.Service
      const effects: string[] = []
      yield* register(tools, stops[0]!, effects)
      const snapshot = yield* tools.snapshot()
      const exit = yield* execute(
        snapshot,
        `
        try { await Promise.allSettled([tools.leaf({})]); }
        finally { await tools.later({}); }
        return "unsafe success";
      `,
      ).pipe(Effect.exit)
      expect({ effects, exit: exit._tag }).toEqual({ effects: ["leaf"], exit: "Failure" })
    }),
  )

  it.effect("keeps ordinary Tool.Error and corrected permission failures catchable", () =>
    Effect.gen(function* () {
      const tools = yield* Tool.Service
      const effects: string[] = []
      yield* tools.transform((editor) => {
        editor.add(leaf(() => Effect.fail(new Tool.Error({ message: "known rejection" }))))
        editor.add({
          ...leaf(() =>
            Effect.fail(
              new Tool.Error({
                message: "Use another tool",
                error: new Permission.CorrectedError({ feedback: "Use another tool" }),
              }),
            ),
          ),
          name: "corrected",
        })
        editor.add({
          ...leaf(() =>
            Effect.sync(() => {
              effects.push("later")
              return { output: "done" }
            }),
          ),
          name: "later",
        })
      })
      const result = yield* execute(
        yield* tools.snapshot(),
        `
        try { await tools.leaf({}); } catch (error) {}
        try { await tools.corrected({}); } catch (error) {}
        return await tools.later({});
      `,
      )
      expect(effects).toEqual(["later"])
      expect(result.output).toMatchObject({ output: "done" })
    }),
  )

  it.effect("does not leak a stopped invocation into the next execution of the same snapshot", () =>
    Effect.gen(function* () {
      const tools = yield* Tool.Service
      const effects: string[] = []
      yield* register(tools, stops[0]!, effects)
      const snapshot = yield* tools.snapshot()
      const first = yield* execute(snapshot, retryProgram).pipe(Effect.exit)
      const second = yield* execute(snapshot, `return await tools.later({});`).pipe(Effect.exit)
      expect({ effects, first: first._tag, second: second._tag }).toEqual({
        effects: ["leaf", "later"],
        first: "Failure",
        second: "Success",
      })
    }),
  )

  it.effect("blocks fetch after an unawaited native defect while a started sibling completes", () =>
    Effect.gen(function* () {
      const tools = yield* Tool.Service
      const observed = yield* Deferred.make<void>()
      let effects = 0
      yield* tools.transform((editor) => {
        editor.add(
          leaf(() =>
            Effect.sync(() => {
              effects++
            }).pipe(
              Effect.andThen(Effect.die("background host defect")),
              Effect.ensuring(Deferred.succeed(observed, undefined)),
            ),
          ),
        )
        editor.add({
          ...leaf(() =>
            Deferred.await(observed).pipe(Effect.andThen(Effect.yieldNow), Effect.as({ output: "witness" })),
          ),
          name: "witness",
        })
      })
      const fetch = yield* Effect.acquireRelease(
        Effect.sync(() => spyOn(globalThis, "fetch").mockResolvedValue(new Response("offline fixture"))),
        (fetch) => Effect.sync(() => fetch.mockRestore()),
      )
      const exit = yield* execute(
        yield* tools.snapshot(),
        `
        tools.leaf({});
        await tools.witness({});
        try { await fetch("https://fixture.invalid/background", { method: "POST" }); } catch (error) {}
        return "unsafe success";
      `,
      ).pipe(Effect.exit)
      expect({ effects, fetches: fetch.mock.calls.length, exit: exit._tag }).toEqual({
        effects: 1,
        fetches: 0,
        exit: "Failure",
      })
      if (Exit.isFailure(exit)) expect(Cause.pretty(exit.cause)).toContain("background host defect")
    }),
  )

  it.effect("does not allow fetch from a catch or finally after a native stop", () =>
    Effect.gen(function* () {
      const tools = yield* Tool.Service
      const effects: string[] = []
      yield* register(tools, stops[0]!, effects)
      const fetch = yield* Effect.acquireRelease(
        Effect.sync(() => spyOn(globalThis, "fetch").mockResolvedValue(new Response("offline fixture"))),
        (fetch) => Effect.sync(() => fetch.mockRestore()),
      )
      const snapshot = yield* tools.snapshot()
      const exit = yield* execute(
        snapshot,
        `
        try { await tools.leaf({}); }
        catch (error) { await fetch("https://fixture.invalid/catch", { method: "POST" }); }
        finally { await fetch("https://fixture.invalid/finally", { method: "POST" }); }
        return "unsafe success";
      `,
      ).pipe(Effect.exit)
      expect({ effects, fetches: fetch.mock.calls.length, exit: exit._tag }).toEqual({
        effects: ["leaf"],
        fetches: 0,
        exit: "Failure",
      })
    }),
  )
})

const testLLM = TestLLM.layer({ fallback: [] })
const model = LanguageModel.make({ id: "codemode-safety", provider: "fake", route: OpenAIChat.route })
const runnerLayer = AppNodeBuilder.build(
  LayerNode.group([
    Agent.node,
    Database.node,
    SessionRunnerLLM.node,
    SessionProjector.node,
    SessionStore.node,
    SessionInbox.node,
    Tool.node,
  ]),
  [
    Bus.node.replace(Bus.configured({ persist: true })),
    Snapshot.node.replace(Snapshot.noopLayer),
    LayerNodePlatform.llmClient.replace(TestLLM.clientLayer.pipe(Layer.provide(testLLM))),
    SessionRunnerModel.node.replace(
      Layer.mock(SessionRunnerModel.Service, {
        resolve: () =>
          Effect.succeed(
            SessionRunnerModel.resolved(model, {
              capabilities: { tools: true, input: ["text"], output: ["text"] },
              cost: [],
              limit: { context: 200_000, output: 32_000 },
            }),
          ),
      }),
    ),
    InstructionBuiltIns.node.replace(
      Layer.mock(InstructionBuiltIns.Service, { load: () => Effect.succeed(Instructions.empty) }),
    ),
    InstructionDiscovery.node.replace(
      Layer.mock(InstructionDiscovery.Service, {
        project: true,
        global: true,
        load: () => Effect.succeed(Instructions.empty),
      }),
    ),
    SkillInstructions.node.replace(
      Layer.mock(SkillInstructions.Service, { load: () => Effect.succeed(Instructions.empty) }),
    ),
    ReferenceInstructions.node.replace(
      Layer.mock(ReferenceInstructions.Service, { load: () => Effect.succeed(Instructions.empty) }),
    ),
    McpInstructions.node.replace(
      Layer.mock(McpInstructions.Service, { load: () => Effect.succeed(Instructions.empty) }),
    ),
    Location.node.replace(Location.boundNode({ directory: AbsolutePath.make("/codemode-safety") })),
    Permission.node.replace(permissionLayer()),
    Config.node.replace(Config.testLayer([])),
    PluginSupervisor.node.replace(Layer.empty),
    Plugin.node.replace(Layer.mock(Plugin.Service, { awaitActivation: Effect.void })),
    SessionModelTransport.node.replace(
      Layer.mock(SessionModelTransport.Service, {
        bind: () => ({ execute: () => Effect.die("Unexpected WebSocket execution") }),
        close: () => Effect.void,
        closeAll: Effect.void,
      }),
    ),
    Model.node.replace(
      Layer.mock(Model.Service, {
        get: () => Effect.undefined,
        all: () => Effect.succeed([]),
        available: () => Effect.succeed([]),
        default: () => Effect.undefined,
        small: () => Effect.undefined,
      }),
    ),
  ],
).pipe(Layer.provideMerge(testLLM))
const runnerIt = testEffect(runnerLayer)
const setupRunner = Effect.gen(function* () {
  const db = (yield* Database.Service).db
  const agents = yield* Agent.Service
  yield* agents.transform((editor) =>
    editor.update(Agent.ID.make("build"), (agent) => {
      agent.mode = "primary"
    }),
  )
  const inbox = yield* SessionInbox.Service
  const runner = yield* SessionRunner.Service
  const store = yield* SessionStore.Service
  const tools = yield* Tool.Service
  const llm = yield* TestLLM.Service
  yield* db
    .insert(ProjectTable)
    .values({ id: Project.ID.global, worktree: AbsolutePath.make("/codemode-safety"), sandboxes: [] })
    .onConflictDoNothing()
    .run()
    .pipe(Effect.orDie)
  yield* db
    .insert(SessionTable)
    .values({
      id: sessionID,
      project_id: Project.ID.global,
      slug: "codemode-safety",
      directory: "/codemode-safety",
      title: "test",
      version: "test",
    })
    .run()
    .pipe(Effect.orDie)
  yield* inbox.admit({
    id: SessionMessage.ID.make("msg_codemode_input"),
    sessionID,
    item: { type: "user", delivery: "queue", payload: { text: "Call the Code Mode leaf" } },
  })
  return { db, runner, store, tools, llm }
})

describe("default Code Mode durable runner stop", () => {
  for (const stop of stops) {
    runnerIt.effect(`settles ${stop.name} without another provider request`, () =>
      Effect.gen(function* () {
        const s = yield* setupRunner
        const effects: string[] = []
        yield* register(s.tools, stop, effects)
        yield* s.llm.push(
          TestLLM.tool("call-execute", "execute", { code: retryProgram }),
          TestLLM.text("Unsafe provider continuation", "later-text"),
        )
        const exit = yield* s.runner.drain({ sessionID, force: false }).pipe(Effect.exit)
        if (s.llm.requests.length === 0 && Exit.isFailure(exit)) return yield* Effect.failCause(exit.cause)
        expect({ effects, requests: s.llm.requests.length, exit: exit._tag }).toEqual({
          effects: stop.effects ? ["leaf"] : [],
          requests: 1,
          exit: "Failure",
        })
        if (Exit.isFailure(exit)) expect(Cause.hasInterruptsOnly(exit.cause)).toBe(!stop.effects)
        const messages = yield* s.store.context(sessionID)
        const assistant = messages.find((message) => message.type === "assistant")
        expect(assistant?.content).toMatchObject([
          Expected.failedTool(
            { id: "call-execute" },
            {
              error: {
                type: stop.effects
                  ? "tool.outcome-unknown"
                  : stop.decision === blocked
                    ? "permission.rejected"
                    : "aborted",
              },
            },
          ),
        ])
        const events = yield* s.db.select().from(EventTable).orderBy(asc(EventTable.seq)).all().pipe(Effect.orDie)
        expect(
          events
            .map((event) => event.type)
            .filter(
              (type) =>
                type.startsWith("session.tool.") ||
                (type.startsWith("session.step.") && type !== "session.step.streamed.1"),
            ),
        ).toEqual([
          "session.step.started.1",
          "session.tool.input.started.1",
          "session.tool.input.ended.1",
          "session.tool.called.1",
          "session.tool.failed.2",
          "session.step.failed.1",
        ])
        if (stop.effects) {
          const again = yield* s.runner.drain({ sessionID, force: true }).pipe(Effect.exit)
          expect(again._tag).toBe("Failure")
          expect({ effects, requests: s.llm.requests.length }).toEqual({ effects: ["leaf"], requests: 1 })
        }
      }),
    )
  }

  runnerIt.effect("settles interruption of a pending Code Mode leaf without continuation", () =>
    Effect.gen(function* () {
      const s = yield* setupRunner
      const started = yield* Deferred.make<void>()
      let cleaned = 0
      yield* s.tools.transform((editor) =>
        editor.add(
          leaf(() =>
            Deferred.succeed(started, undefined).pipe(
              Effect.andThen(Effect.never),
              Effect.ensuring(Effect.sync(() => cleaned++)),
            ),
          ),
        ),
      )
      yield* s.llm.push(
        TestLLM.tool("call-execute", "execute", { code: `await tools.leaf({}); return "unsafe success";` }),
        TestLLM.text("Unsafe continuation", "later-text"),
      )
      const run = yield* s.runner.drain({ sessionID, force: false }).pipe(Effect.forkChild)
      yield* Deferred.await(started).pipe(Effect.raceFirst(Fiber.join(run)))
      yield* Fiber.interrupt(run)
      const exit = yield* Fiber.await(run)
      expect(Exit.isFailure(exit) && Cause.hasInterruptsOnly(exit.cause)).toBe(true)
      expect({ cleaned, requests: s.llm.requests.length }).toEqual({ cleaned: 1, requests: 1 })
      const messages = yield* s.store.context(sessionID)
      expect(messages.find((message) => message.type === "assistant")?.content).toMatchObject([
        Expected.failedTool({ id: "call-execute" }, { error: { type: "tool.outcome-unknown" } }),
      ])
    }),
  )
})
