import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { InstanceState } from "@/effect/instance-state"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Runner } from "@/effect/runner"
import { BackgroundJob } from "@/background/job"
import { Effect, Latch, Layer, Scope, Context } from "effect"
import { Session } from "./session"
import { SessionID } from "./schema"
import { SessionStatus } from "./status"

export interface Interface {
  readonly assertNotBusy: (sessionID: SessionID) => Effect.Effect<void, Session.BusyError>
  readonly cancel: (sessionID: SessionID) => Effect.Effect<void>
  readonly cancelProbe: (sessionID: SessionID) => Effect.Effect<Effect.Effect<boolean>>
  readonly wasCancelled: (sessionID: SessionID) => Effect.Effect<boolean>
  readonly ensureRunning: (
    sessionID: SessionID,
    onInterrupt: Effect.Effect<SessionV1.WithParts>,
    work: Effect.Effect<SessionV1.WithParts>,
  ) => Effect.Effect<SessionV1.WithParts>
  readonly startShell: (
    sessionID: SessionID,
    onInterrupt: Effect.Effect<SessionV1.WithParts>,
    work: Effect.Effect<SessionV1.WithParts>,
    ready?: Latch.Latch,
  ) => Effect.Effect<SessionV1.WithParts, Session.BusyError>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionRunState") {}

type CancelProbe = {
  cancelled: boolean
}

type CancelProbes = {
  active?: CancelProbe
  queued?: CancelProbe
}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const background = yield* BackgroundJob.Service
    const status = yield* SessionStatus.Service

    const state = yield* InstanceState.make(
      Effect.fn("SessionRunState.state")(function* () {
        const scope = yield* Scope.Scope
        const runners = new Map<SessionID, Runner.Runner<SessionV1.WithParts>>()
        const cancelled = new Map<SessionID, CancelProbes>()
        yield* Effect.addFinalizer(
          Effect.fnUntraced(function* () {
            yield* Effect.forEach(runners.values(), (runner) => runner.cancel, {
              concurrency: "unbounded",
              discard: true,
            })
            runners.clear()
            cancelled.clear()
          }),
        )
        return { runners, cancelled, scope }
      }),
    )

    const runner = Effect.fn("SessionRunState.runner")(function* (
      sessionID: SessionID,
      onInterrupt: Effect.Effect<SessionV1.WithParts>,
    ) {
      const data = yield* InstanceState.get(state)
      const existing = data.runners.get(sessionID)
      if (existing) return existing
      const next = Runner.make<SessionV1.WithParts>(data.scope, {
        onIdle: Effect.gen(function* () {
          data.runners.delete(sessionID)
          yield* status.set(sessionID, { type: "idle" })
        }),
        onBusy: status.set(sessionID, { type: "busy" }),
        onInterrupt,
      })
      data.runners.set(sessionID, next)
      return next
    })

    const assertNotBusy = Effect.fn("SessionRunState.assertNotBusy")(function* (sessionID: SessionID) {
      const data = yield* InstanceState.get(state)
      const existing = data.runners.get(sessionID)
      if (existing?.busy) yield* busyError(sessionID)
    })

    const cancel = Effect.fn("SessionRunState.cancel")(function* (sessionID: SessionID) {
      const data = yield* InstanceState.get(state)
      const existing = data.runners.get(sessionID)
      if (!existing || !existing.busy) {
        data.cancelled.delete(sessionID)
        yield* cancelBackgroundJobs(background, sessionID)
        yield* status.set(sessionID, { type: "idle" })
        return
      }
      const probes = data.cancelled.get(sessionID) ?? {}
      const probe = probes.active ?? probes.queued ?? { cancelled: false }
      probe.cancelled = true
      probes.active = probe
      data.cancelled.set(sessionID, probes)
      yield* cancelBackgroundJobs(background, sessionID)
      yield* existing.cancel
    })

    const cancelProbe = Effect.fn("SessionRunState.cancelProbe")(function* (sessionID: SessionID) {
      const probe = (yield* InstanceState.get(state)).cancelled.get(sessionID)?.active
      if (!probe) return Effect.succeed(false)
      return Effect.sync(() => probe.cancelled)
    })

    const wasCancelled = Effect.fn("SessionRunState.wasCancelled")(function* (sessionID: SessionID) {
      const probes = (yield* InstanceState.get(state)).cancelled.get(sessionID)
      return probes?.active?.cancelled ?? probes?.queued?.cancelled ?? false
    })

    const ensureRunning = Effect.fn("SessionRunState.ensureRunning")(function* (
      sessionID: SessionID,
      onInterrupt: Effect.Effect<SessionV1.WithParts>,
      work: Effect.Effect<SessionV1.WithParts>,
    ) {
      const data = yield* InstanceState.get(state)
      const current = yield* runner(sessionID, onInterrupt)
      switch (current.state._tag) {
        case "Idle": {
          const probe = installActiveProbe(data.cancelled, sessionID)
          return yield* current.ensureRunning(
            work.pipe(Effect.ensuring(clearActiveProbe(data.cancelled, sessionID, probe))),
          )
        }
        case "Shell": {
          const probe = installQueuedProbe(data.cancelled, sessionID)
          return yield* current.ensureRunning(activateQueuedProbe(data.cancelled, sessionID, probe, work))
        }
        case "Running":
        case "ShellThenRun":
          return yield* current.ensureRunning(work)
      }
    })

    const startShell = Effect.fn("SessionRunState.startShell")(function* (
      sessionID: SessionID,
      onInterrupt: Effect.Effect<SessionV1.WithParts>,
      work: Effect.Effect<SessionV1.WithParts>,
      ready?: Latch.Latch,
    ) {
      const data = yield* InstanceState.get(state)
      const current = yield* runner(sessionID, onInterrupt)
      if (current.state._tag !== "Idle") {
        return yield* current
          .startShell(work, ready)
          .pipe(Effect.catchTag("RunnerBusy", () => Effect.fail(busyError(sessionID))))
      }
      const probe = installActiveProbe(data.cancelled, sessionID)
      return yield* current
        .startShell(work.pipe(Effect.ensuring(clearActiveProbe(data.cancelled, sessionID, probe))), ready)
        .pipe(Effect.catchTag("RunnerBusy", () => Effect.fail(busyError(sessionID))))
    })

    return Service.of({ assertNotBusy, cancel, cancelProbe, wasCancelled, ensureRunning, startShell })
  }),
)

const cancelBackgroundJobs = Effect.fn("SessionRunState.cancelBackgroundJobs")(function* (
  background: BackgroundJob.Interface,
  sessionID: SessionID,
) {
  const jobs = yield* background.list()
  const pending = new Set<string>([sessionID])
  const cancelled = new Set<string>()
  const matches = (job: BackgroundJob.Info) => {
    if (job.status !== "running") return false
    if (cancelled.has(job.id)) return false
    if (pending.has(job.id)) return true
    if (typeof job.metadata?.sessionId === "string" && pending.has(job.metadata.sessionId)) return true
    return typeof job.metadata?.parentSessionId === "string" && pending.has(job.metadata.parentSessionId)
  }
  let batch = jobs.filter(matches)
  while (batch.length > 0) {
    yield* Effect.forEach(
      batch,
      (job) =>
        background.cancel(job.id).pipe(
          Effect.tap(() =>
            Effect.sync(() => {
              cancelled.add(job.id)
              pending.add(job.id)
              if (typeof job.metadata?.sessionId === "string") pending.add(job.metadata.sessionId)
            }),
          ),
        ),
      { concurrency: "unbounded", discard: true },
    )
    batch = jobs.filter(matches)
  }
})

function busyError(sessionID: SessionID) {
  return new Session.BusyError({ sessionID })
}

function installActiveProbe(probes: Map<SessionID, CancelProbes>, sessionID: SessionID) {
  const probe = { cancelled: false }
  probes.set(sessionID, { active: probe })
  return probe
}

function installQueuedProbe(probes: Map<SessionID, CancelProbes>, sessionID: SessionID) {
  const probe = { cancelled: false }
  const current = probes.get(sessionID) ?? {}
  current.queued = probe
  probes.set(sessionID, current)
  return probe
}

function activateQueuedProbe(
  probes: Map<SessionID, CancelProbes>,
  sessionID: SessionID,
  probe: CancelProbe,
  work: Effect.Effect<SessionV1.WithParts>,
) {
  return Effect.gen(function* () {
    const current = probes.get(sessionID) ?? {}
    current.active = probe
    if (current.queued === probe) current.queued = undefined
    probes.set(sessionID, current)
    return yield* work.pipe(Effect.ensuring(clearActiveProbe(probes, sessionID, probe)))
  })
}

function clearActiveProbe(probes: Map<SessionID, CancelProbes>, sessionID: SessionID, probe: CancelProbe) {
  return Effect.sync(() => {
    const current = probes.get(sessionID)
    if (!current || current.active !== probe) return
    current.active = undefined
    if (current.queued) return
    probes.delete(sessionID)
  })
}

export const node = LayerNode.make({ service: Service, layer: layer, deps: [BackgroundJob.node, SessionStatus.node] })

export * as SessionRunState from "./run-state"
