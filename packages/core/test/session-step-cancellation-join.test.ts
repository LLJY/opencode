import { expect } from "bun:test"
import { LanguageModel, LLM, LLMEvent } from "@opencode/ai"
import { OpenAIChat } from "@opencode/ai/protocols/openai-chat"
import { TestLLM } from "@opencode/ai/testing"
import { Agent } from "@opencode/core/agent"
import { Bus } from "@opencode/core/bus"
import { Database } from "@opencode/core/database/database"
import { AppNodeBuilder } from "@opencode/core/effect/app-node-builder"
import { EventTable } from "@opencode/core/event/sql"
import { Project } from "@opencode/core/project"
import { ProjectTable } from "@opencode/core/project/sql"
import { AbsolutePath } from "@opencode/core/schema"
import { Session } from "@opencode/core/session"
import { SessionEvent } from "@opencode/core/session/event"
import { SessionMessage } from "@opencode/core/session/message"
import { SessionProjector } from "@opencode/core/session/projector"
import { SessionRunnerModel } from "@opencode/core/session/runner/model"
import { SessionStep } from "@opencode/core/session/runner/step"
import { SessionMessageTable, SessionTable } from "@opencode/core/session/sql"
import { Snapshot } from "@opencode/core/snapshot"
import { ToolOutput } from "@opencode/core/tool-output"
import { LayerNode } from "@opencode/util/effect/layer-node"
import { asc, eq } from "drizzle-orm"
import { Cause, Deferred, Effect, Exit, Fiber, Layer } from "effect"
import { testEffect } from "./lib/effect"

const it = testEffect(
  Layer.merge(
    AppNodeBuilder.build(LayerNode.group([Database.node, Bus.node, SessionProjector.node, ToolOutput.node]), [
      Bus.node.replace(Bus.configured({ persist: true })),
    ]),
    TestLLM.testLayer(),
  ),
)

for (const defect of [true, false]) {
  it.effect(`cancellation joining tools retains ${defect ? "completed sibling defect" : "pure interruption"}`, () =>
    Effect.gen(function* () {
      const db = (yield* Database.Service).db
      const bus = yield* Bus.Service
      const llm = yield* TestLLM.Test
      const completed = yield* Deferred.make<void>()
      const pending = yield* Deferred.make<void>()
      const streamed = yield* Deferred.make<void>()
      let cleaned = false
      let executions = 0
      let retries = 0
      const observed: Bus.Interface = {
        ...bus,
        publish: (definition, data, options) =>
          bus
            .publish(definition, data, options)
            .pipe(
              Effect.tap(() =>
                definition.type === SessionEvent.Step.Streamed.type
                  ? Deferred.succeed(streamed, undefined)
                  : Effect.void,
              ),
            ),
      }
      const steps = yield* SessionStep.make.pipe(
        Effect.provideService(Bus.Service, observed),
        Effect.provide(Layer.mock(Snapshot.Service)({ capture: () => Effect.succeed(undefined) })),
      )
      const sessionID = Session.ID.create()
      const assistantMessageID = SessionMessage.ID.create()
      yield* db
        .insert(ProjectTable)
        .values({ id: Project.ID.global, worktree: AbsolutePath.make("/project"), sandboxes: [] })
        .run()
      yield* db
        .insert(SessionTable)
        .values({
          id: sessionID,
          project_id: Project.ID.global,
          slug: "join-cancel",
          directory: "/project",
          version: "test",
        })
        .run()
      const model = SessionRunnerModel.resolved(
        LanguageModel.make({ id: "test", provider: "test", route: OpenAIChat.route }),
        {
          capabilities: { tools: true, input: ["text"], output: ["text"] },
          limit: { context: 100_000, output: 1_000 },
          cost: [],
        },
      )
      yield* llm.push(
        TestLLM.complete(
          { reason: { normalized: "tool-calls" } },
          LLMEvent.toolCall({ id: "completed", name: "first", input: {} }),
          LLMEvent.toolCall({ id: "pending", name: "second", input: {} }),
        ),
      )
      const fiber = yield* steps
        .attempt({
          isLocationClosed: () => false,
          sessionID,
          assistantMessageID,
          agent: Agent.defaultID,
          model,
          prepared: {
            retry: () => Effect.void,
            request: LLM.request({ model: model.model, prompt: "Run siblings" }),
            options: {},
            executeTool: ({ call }) =>
              Effect.gen(function* () {
                executions++
                if (call.id === "completed")
                  return yield* (
                    defect
                      ? Effect.die("completed sibling defect")
                      : Effect.succeed({ content: [{ type: "text" as const, text: "done" }] })
                  ).pipe(Effect.ensuring(Deferred.succeed(completed, undefined)))
                yield* Deferred.await(completed)
                yield* Deferred.succeed(pending, undefined)
                return yield* Effect.never.pipe(
                  Effect.ensuring(
                    Effect.sync(() => {
                      cleaned = true
                    }),
                  ),
                )
              }),
          },
          retry: () =>
            Effect.sync(() => {
              retries++
              return { retry: false } as const
            }),
          recoverContinuation: true,
          recoverOverflow: Effect.succeed(false),
        })
        .pipe(Effect.forkChild)
      yield* Deferred.await(completed)
      yield* Deferred.await(pending)
      yield* Deferred.await(streamed)
      yield* Effect.yieldNow
      yield* Fiber.interrupt(fiber)
      const exit = yield* Fiber.await(fiber)
      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isSuccess(exit)) throw new Error("cancelled attempt succeeded")
      expect(exit.cause.reasons.filter(Cause.isDieReason).map((reason) => reason.defect)).toEqual(
        defect ? ["completed sibling defect"] : [],
      )
      expect(Cause.hasInterrupts(exit.cause)).toBe(true)
      expect(cleaned).toBe(true)
      expect(executions).toBe(2)
      expect(retries).toBe(0)
      expect(yield* llm.requests()).toHaveLength(1)
      const message = yield* db
        .select()
        .from(SessionMessageTable)
        .where(eq(SessionMessageTable.id, assistantMessageID))
        .get()
      expect(message?.data).toMatchObject({
        error: defect
          ? { type: "unknown", message: "completed sibling defect" }
          : { type: "aborted", message: "Step interrupted" },
      })
      const events = yield* db
        .select({ type: EventTable.type })
        .from(EventTable)
        .where(eq(EventTable.aggregate_id, sessionID))
        .orderBy(asc(EventTable.seq))
        .all()
      const types = events.map((event) => event.type)
      expect(types.filter((type) => type === "session.step.failed.1")).toHaveLength(1)
      expect(types).not.toContain("session.step.ended.1")
      expect(types.lastIndexOf("session.tool.failed.2")).toBeLessThan(types.indexOf("session.step.failed.1"))
    }),
  )
}
