import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { describe, expect, test } from "bun:test"
import { Effect, Fiber, Layer } from "effect"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Permission } from "@/permission"
import { InstanceBootstrap } from "@/project/bootstrap"
import { InstanceStore } from "@/project/instance-store"
import { LLM } from "@/session/llm"
import { testEffect, pollWithTimeout } from "../lib/effect"

const noopBootstrap = Layer.succeed(InstanceBootstrap.Service, InstanceBootstrap.Service.of({ run: Effect.void }))
const env = AppNodeBuilder.build(
  LayerNode.group([Permission.node, EventV2Bridge.node, CrossSpawnSpawner.node, InstanceStore.node]),
  [[InstanceStore.bootstrapNode, noopBootstrap]],
)
const it = testEffect(env)

const tools = [{ name: "workflow_tool", args: '{"title":"Review changes"}' }]

const pendingFor = (sessionID: string) =>
  Effect.gen(function* () {
    const permission = yield* Permission.Service
    return yield* pollWithTimeout(
      permission.list().pipe(Effect.map((requests) => requests.find((request) => request.sessionID === sessionID))),
      `timed out waiting for ${sessionID} workflow approval`,
    )
  })

const cleared = (id: PermissionV1.ID) =>
  Effect.gen(function* () {
    const permission = yield* Permission.Service
    yield* pollWithTimeout(
      permission.list().pipe(Effect.map((requests) => (requests.some((request) => request.id === id) ? undefined : true))),
      `timed out waiting for ${id} workflow approval to clear`,
    )
  })

describe("session.llm workflow preapproval", () => {
  const rule = (permission: string, action: PermissionV1.Rule["action"], pattern = "*") => ({
    permission,
    pattern,
    action,
  })

  test("preapproves only an explicit allow", () => {
    expect(
      LLM.workflowPreapprovedTools(
        ["allowed", "asked", "denied", "unmatched"],
        [rule("allowed", "allow"), rule("asked", "ask"), rule("denied", "deny")],
      ),
    ).toEqual(["allowed"])
  })

  test("does not preapprove a tool whose allow is scoped to a pattern", () => {
    // The workflow decides before any arguments exist, so an allow that only covers
    // `git status` cannot stand in for every call the model might make.
    expect(LLM.workflowPreapprovedTools(["bash"], [rule("bash", "allow", "git status")])).toEqual([])
    expect(LLM.workflowPreapprovedTools(["bash"], [rule("bash", "allow", "*")])).toEqual(["bash"])
  })

  test("resolves the permission alias each tool asks with", () => {
    expect(
      LLM.workflowPreapprovedTools(["write", "apply_patch", "read_mcp_resource"], [rule("edit", "allow")]),
    ).toEqual(["write", "apply_patch"])
    expect(LLM.workflowPreapprovedTools(["read_mcp_resource"], [rule("read", "allow")])).toEqual([
      "read_mcp_resource",
    ])
  })

  test("keeps the last matching rule authoritative", () => {
    expect(LLM.workflowPreapprovedTools(["bash"], [rule("*", "allow"), rule("bash", "deny")])).toEqual([])
    expect(LLM.workflowPreapprovedTools(["bash"], [rule("bash", "deny"), rule("*", "allow")])).toEqual(["bash"])
  })

  test("preapproves nothing without a ruleset", () => {
    expect(LLM.workflowPreapprovedTools(["bash", "edit", "lookup"], [])).toEqual([])
  })
})

describe("session.llm workflow approval", () => {
  it.instance("aborting removes the pending workflow permission and rejects approval", () =>
    Effect.gen(function* () {
      const controller = new AbortController()
      const permission = yield* Permission.Service
      const approval = yield* LLM.waitForWorkflowToolApproval({
        permission,
        abort: controller.signal,
        sessionID: "session-a",
        tools,
      }).pipe(Effect.forkScoped)

      const pending = yield* pendingFor("session-a")
      controller.abort()

      expect(yield* Fiber.join(approval)).toEqual({ approved: false })
      yield* cleared(pending.id)
    }),
  )

  it.instance("aborting one workflow approval leaves another session pending and resolvable", () =>
    Effect.gen(function* () {
      const controllerA = new AbortController()
      const controllerB = new AbortController()
      const permission = yield* Permission.Service
      const approvalA = yield* LLM.waitForWorkflowToolApproval({
        permission,
        abort: controllerA.signal,
        sessionID: "session-a",
        tools,
      }).pipe(Effect.forkScoped)
      const approvalB = yield* LLM.waitForWorkflowToolApproval({
        permission,
        abort: controllerB.signal,
        sessionID: "session-b",
        tools,
      }).pipe(Effect.forkScoped)

      const pendingA = yield* pendingFor("session-a")
      const pendingB = yield* pendingFor("session-b")
      controllerA.abort()

      expect(yield* Fiber.join(approvalA)).toEqual({ approved: false })
      yield* cleared(pendingA.id)
      expect((yield* permission.list()).map((request) => request.id)).toEqual([pendingB.id])

      yield* permission.reply({ requestID: pendingB.id, reply: "once" })
      expect(yield* Fiber.join(approvalB)).toEqual({ approved: true })
      yield* cleared(pendingB.id)
    }),
  )

  it.instance("approves workflow tools after a permission reply", () =>
    Effect.gen(function* () {
      const controller = new AbortController()
      const permission = yield* Permission.Service
      const approval = yield* LLM.waitForWorkflowToolApproval({
        permission,
        abort: controller.signal,
        sessionID: "session-a",
        tools,
      }).pipe(Effect.forkScoped)

      const pending = yield* pendingFor("session-a")
      yield* permission.reply({ requestID: pending.id, reply: "once" })

      expect(yield* Fiber.join(approval)).toEqual({ approved: true })
      yield* cleared(pending.id)
    }),
  )

  it.instance("rejects an already-aborted workflow approval without creating a permission", () =>
    Effect.gen(function* () {
      const controller = new AbortController()
      controller.abort()

      const permission = yield* Permission.Service
      expect(
        yield* LLM.waitForWorkflowToolApproval({
          permission,
          abort: controller.signal,
          sessionID: "session-a",
          tools,
        }),
      ).toEqual({ approved: false })

      expect(yield* permission.list()).toEqual([])
    }),
  )
})
