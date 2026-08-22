import { describe, expect } from "bun:test"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { MessageID, SessionID } from "@/session/schema"
import { SessionRunState } from "@/session/run-state"
import { awaitWithTimeout, testEffect } from "../lib/effect"

const it = testEffect(LayerNode.compile(SessionRunState.node))

describe("SessionRunState", () => {
  it.instance(
    "isolates cancel probes between cancelled and replacement runs",
    Effect.gen(function* () {
      const run = yield* SessionRunState.Service
      const sessionID = SessionID.make("ses_cancel_probe_race")
      const output = assistant(sessionID)
      const firstProbe = yield* Deferred.make<Effect.Effect<boolean>>()
      const firstInterrupted = yield* Deferred.make<void>()
      const holdFirstCleanup = yield* Deferred.make<void>()
      const secondProbe = yield* Deferred.make<Effect.Effect<boolean>>()
      const releaseSecond = yield* Deferred.make<void>()

      yield* Effect.gen(function* () {
        const first = yield* run
          .ensureRunning(
            sessionID,
            Effect.succeed(output),
            Effect.gen(function* () {
              yield* Deferred.succeed(firstProbe, yield* run.cancelProbe(sessionID))
              return yield* Effect.never
            }).pipe(
              Effect.onInterrupt(() => Deferred.succeed(firstInterrupted, undefined)),
              Effect.ensuring(Deferred.await(holdFirstCleanup)),
              Effect.as(output),
            ),
          )
          .pipe(Effect.forkChild)

        const oldWasCancelled = yield* awaitWithTimeout(
          Deferred.await(firstProbe),
          "first run did not capture cancel probe",
        )
        expect(yield* oldWasCancelled).toBe(false)

        const cancel = yield* run.cancel(sessionID).pipe(Effect.forkChild)
        yield* awaitWithTimeout(Deferred.await(firstInterrupted), "first run was not interrupted")
        expect(yield* oldWasCancelled).toBe(true)

        const second = yield* run
          .ensureRunning(
            sessionID,
            Effect.succeed(output),
            Effect.gen(function* () {
              yield* Deferred.succeed(secondProbe, yield* run.cancelProbe(sessionID))
              yield* Deferred.await(releaseSecond)
              return output
            }),
          )
          .pipe(Effect.forkChild)
        const newWasCancelled = yield* awaitWithTimeout(
          Deferred.await(secondProbe),
          "second run did not capture cancel probe",
        )

        expect(yield* newWasCancelled).toBe(false)
        expect(yield* run.wasCancelled(sessionID)).toBe(false)

        yield* Deferred.succeed(holdFirstCleanup, undefined)
        const cancelExit = yield* awaitWithTimeout(Fiber.await(cancel), "cancel did not finish after cleanup release")
        expect(Exit.isSuccess(cancelExit)).toBe(true)
        expect(yield* newWasCancelled).toBe(false)

        yield* Deferred.succeed(releaseSecond, undefined)
        expect(yield* Fiber.join(second)).toBe(output)
        expect(yield* Fiber.join(first)).toBe(output)
        expect(yield* run.wasCancelled(sessionID)).toBe(false)
      }).pipe(
        Effect.ensuring(
          Effect.all([Deferred.succeed(holdFirstCleanup, undefined), Deferred.succeed(releaseSecond, undefined)], {
            discard: true,
          }).pipe(Effect.ignore),
        ),
      )
    }),
    3_000,
  )

  it.instance(
    "preserves cancellation when direct runs are cancelled before probe capture",
    Effect.gen(function* () {
      const run = yield* SessionRunState.Service
      const sessionID = SessionID.make("ses_cancel_probe_before_capture")
      const output = assistant(sessionID)
      const started = yield* Deferred.make<void>()
      const interruptedProbe = yield* Deferred.make<Effect.Effect<boolean>>()
      const holdCleanup = yield* Deferred.make<void>()

      yield* Effect.gen(function* () {
        const fiber = yield* run
          .ensureRunning(
            sessionID,
            Effect.succeed(output),
            Effect.gen(function* () {
              yield* Deferred.succeed(started, undefined)
              return yield* Effect.never
            }).pipe(
              Effect.ensuring(
                Effect.gen(function* () {
                  yield* Deferred.succeed(interruptedProbe, yield* run.cancelProbe(sessionID))
                  yield* Deferred.await(holdCleanup)
                }),
              ),
              Effect.as(output),
            ),
          )
          .pipe(Effect.forkChild)

        yield* awaitWithTimeout(Deferred.await(started), "run did not start")
        const cancel = yield* run.cancel(sessionID).pipe(Effect.forkChild)
        const wasCancelled = yield* awaitWithTimeout(
          Deferred.await(interruptedProbe),
          "interrupted run did not capture cancel probe",
        )

        expect(yield* wasCancelled).toBe(true)
        yield* Deferred.succeed(holdCleanup, undefined)
        expect(Exit.isSuccess(yield* Fiber.await(cancel))).toBe(true)
        expect(yield* Fiber.join(fiber)).toBe(output)
        expect(yield* run.wasCancelled(sessionID)).toBe(false)
      }).pipe(Effect.ensuring(Deferred.succeed(holdCleanup, undefined).pipe(Effect.ignore)))
    }),
    3_000,
  )

  it.instance(
    "installs a fresh cancel probe for runs queued behind shells",
    Effect.gen(function* () {
      const run = yield* SessionRunState.Service
      const sessionID = SessionID.make("ses_cancel_probe_shell_queue")
      const output = assistant(sessionID)
      const shellStarted = yield* Deferred.make<void>()
      const releaseShell = yield* Deferred.make<void>()
      const queuedProbe = yield* Deferred.make<Effect.Effect<boolean>>()
      const releaseQueued = yield* Deferred.make<void>()

      yield* Effect.gen(function* () {
        const shell = yield* run
          .startShell(
            sessionID,
            Effect.succeed(output),
            Effect.gen(function* () {
              yield* Deferred.succeed(shellStarted, undefined)
              yield* Deferred.await(releaseShell)
              return output
            }),
          )
          .pipe(Effect.forkChild)

        yield* awaitWithTimeout(Deferred.await(shellStarted), "shell did not start")

        const queued = yield* run
          .ensureRunning(
            sessionID,
            Effect.succeed(output),
            Effect.gen(function* () {
              yield* Deferred.succeed(queuedProbe, yield* run.cancelProbe(sessionID))
              yield* Deferred.await(releaseQueued)
              return output
            }),
          )
          .pipe(Effect.forkChild)

        yield* Effect.yieldNow
        expect(yield* Deferred.isDone(queuedProbe)).toBe(false)

        yield* Deferred.succeed(releaseShell, undefined)
        expect(yield* Fiber.join(shell)).toBe(output)

        const queuedWasCancelled = yield* awaitWithTimeout(
          Deferred.await(queuedProbe),
          "queued run did not capture cancel probe",
        )
        expect(yield* queuedWasCancelled).toBe(false)

        yield* run.cancel(sessionID)
        expect(yield* queuedWasCancelled).toBe(true)
        expect(yield* Fiber.join(queued)).toBe(output)
        expect(yield* run.wasCancelled(sessionID)).toBe(false)
        expect(yield* yield* run.cancelProbe(sessionID)).toBe(false)
      }).pipe(
        Effect.ensuring(
          Effect.all([Deferred.succeed(releaseShell, undefined), Deferred.succeed(releaseQueued, undefined)], {
            discard: true,
          }).pipe(Effect.ignore),
        ),
      )
    }),
    3_000,
  )

  it.instance(
    "preserves cancellation when shell-queued runs are cancelled before probe capture",
    Effect.gen(function* () {
      const run = yield* SessionRunState.Service
      const sessionID = SessionID.make("ses_cancel_probe_shell_queue_before_capture")
      const output = assistant(sessionID)
      const shellStarted = yield* Deferred.make<void>()
      const releaseShell = yield* Deferred.make<void>()
      const queuedStarted = yield* Deferred.make<void>()
      const queuedInterruptedProbe = yield* Deferred.make<Effect.Effect<boolean>>()
      const holdQueuedCleanup = yield* Deferred.make<void>()

      yield* Effect.gen(function* () {
        const shell = yield* run
          .startShell(
            sessionID,
            Effect.succeed(output),
            Effect.gen(function* () {
              yield* Deferred.succeed(shellStarted, undefined)
              yield* Deferred.await(releaseShell)
              return output
            }),
          )
          .pipe(Effect.forkChild)

        yield* awaitWithTimeout(Deferred.await(shellStarted), "shell did not start")
        const queued = yield* run
          .ensureRunning(
            sessionID,
            Effect.succeed(output),
            Effect.gen(function* () {
              yield* Deferred.succeed(queuedStarted, undefined)
              return yield* Effect.never
            }).pipe(
              Effect.ensuring(
                Effect.gen(function* () {
                  yield* Deferred.succeed(queuedInterruptedProbe, yield* run.cancelProbe(sessionID))
                  yield* Deferred.await(holdQueuedCleanup)
                }),
              ),
              Effect.as(output),
            ),
          )
          .pipe(Effect.forkChild)

        yield* Deferred.succeed(releaseShell, undefined)
        expect(yield* Fiber.join(shell)).toBe(output)
        yield* awaitWithTimeout(Deferred.await(queuedStarted), "queued run did not start")

        const cancel = yield* run.cancel(sessionID).pipe(Effect.forkChild)
        const wasCancelled = yield* awaitWithTimeout(
          Deferred.await(queuedInterruptedProbe),
          "queued run did not capture cancel probe on interrupt",
        )

        expect(yield* wasCancelled).toBe(true)
        yield* Deferred.succeed(holdQueuedCleanup, undefined)
        expect(Exit.isSuccess(yield* Fiber.await(cancel))).toBe(true)
        expect(yield* Fiber.join(queued)).toBe(output)
        expect(yield* run.wasCancelled(sessionID)).toBe(false)
      }).pipe(
        Effect.ensuring(
          Effect.all([Deferred.succeed(releaseShell, undefined), Deferred.succeed(holdQueuedCleanup, undefined)], {
            discard: true,
          }).pipe(Effect.ignore),
        ),
      )
    }),
    3_000,
  )
})

function assistant(sessionID: SessionID): SessionV1.WithParts {
  return {
    info: {
      id: MessageID.make("msg_cancel_probe_race_assistant"),
      sessionID,
      role: "assistant",
      parentID: MessageID.make("msg_cancel_probe_race_user"),
      time: { created: 0 },
      modelID: ModelV2.ID.make("test-model"),
      providerID: ProviderV2.ID.make("test"),
      mode: "build",
      agent: "build",
      path: { cwd: "/tmp", root: "/tmp" },
      cost: 0,
      tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    },
    parts: [],
  }
}
