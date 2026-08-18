import { describe, expect, test } from "bun:test"
import { GitLabWorkflowLanguageModel } from "gitlab-ai-provider"
import { LLM } from "@/session/llm"
import { tmpdir } from "../fixture/fixture"

type ApprovalHandler = NonNullable<GitLabWorkflowLanguageModel["approvalHandler"]>

type Internals = {
  workflowOptions: { approvalHandler?: ApprovalHandler }
  detectedProjectPath: string | null
  sessionWorkflows: Map<string, { workflowId: string }>
  setSessionWorkflow: (key: string, value: { workflowId: string }) => void
}

const internals = (model: GitLabWorkflowLanguageModel) => model as unknown as Internals

// Two facades over the one memoized vendor model, written in the order a real race
// produces them: session A configures its stream, then session B configures its own
// before A's `doStream` has read anything back.
const interleaved = async (directory: string) => {
  const model = new GitLabWorkflowLanguageModel(
    "duo-workflow",
    {
      provider: "gitlab.workflow",
      instanceUrl: "https://gitlab.example.test",
      getHeaders: () => ({}),
      fetch: Object.assign(async () => new Response(), { preconnect: () => undefined }) satisfies typeof fetch,
    },
    { workingDirectory: directory },
  )
  const a = LLM.createWorkflowModelFacade(model)
  const b = LLM.createWorkflowModelFacade(model)
  const invoked: string[] = []
  const approved: string[] = []

  a.sessionID = "session-a"
  a.systemPrompt = "system-a"
  a.toolExecutor = async () => {
    invoked.push(`a:${a.sessionID}`)
    return { result: "a" }
  }
  a.sessionPreapprovedTools = ["tool-a"]
  a.approvalHandler = async () => {
    approved.push(`a:${a.sessionID}`)
    return { approved: true }
  }

  b.sessionID = "session-b"
  b.systemPrompt = "system-b"
  b.toolExecutor = async () => {
    invoked.push(`b:${b.sessionID}`)
    return { result: "b" }
  }
  b.sessionPreapprovedTools = ["tool-b"]
  b.approvalHandler = async () => {
    approved.push(`b:${b.sessionID}`)
    return { approved: true }
  }

  return { model, a, b, invoked, approved }
}

describe("session.llm workflow model facade", () => {
  test("keeps interleaved per-stream state on the session that wrote it", async () => {
    await using tmp = await tmpdir()
    const { a, b } = await interleaved(tmp.path)

    expect(a).toBeInstanceOf(GitLabWorkflowLanguageModel)
    expect(b).toBeInstanceOf(GitLabWorkflowLanguageModel)
    expect(a.sessionID).toBe("session-a")
    expect(a.systemPrompt).toBe("system-a")
    expect(a.sessionPreapprovedTools).toEqual(["tool-a"])
    expect(b.sessionID).toBe("session-b")
    expect(b.systemPrompt).toBe("system-b")
    expect(b.sessionPreapprovedTools).toEqual(["tool-b"])
  })

  // `doStream` snapshots the executor at entry and the approval path reads the handler
  // live, so both are read here the way the vendor reads them.
  test("dispatches tools and approvals to the session that owns them", async () => {
    await using tmp = await tmpdir()
    const { a, b, invoked, approved } = await interleaved(tmp.path)

    const executorA = a.toolExecutor
    const executorB = b.toolExecutor
    if (!executorA || !executorB) throw new Error("expected workflow tool executors")
    expect(await executorA("tool", "{}", "request-a")).toEqual({ result: "a" })
    expect(await executorB("tool", "{}", "request-b")).toEqual({ result: "b" })
    expect(invoked).toEqual(["a:session-a", "b:session-b"])

    const handlerA = internals(a).workflowOptions.approvalHandler
    const handlerB = internals(b).workflowOptions.approvalHandler
    if (!handlerA || !handlerB) throw new Error("expected workflow approval handlers")
    await handlerA([{ name: "tool-a", args: "{}" }])
    await handlerB([{ name: "tool-b", args: "{}" }])
    expect(approved).toEqual(["a:session-a", "b:session-b"])
  })

  test("leaves the vendor's shared caches on the memoized instance", async () => {
    await using tmp = await tmpdir()
    const { model, a, b } = await interleaved(tmp.path)

    internals(a).detectedProjectPath = "group/project"
    expect(internals(model).detectedProjectPath).toBe("group/project")
    expect(internals(b).detectedProjectPath).toBe("group/project")

    internals(a).setSessionWorkflow("session-a", { workflowId: "workflow-a" })
    expect(internals(model).sessionWorkflows.get("session-a")).toEqual({ workflowId: "workflow-a" })
  })

  test("never writes per-stream state onto the memoized instance", async () => {
    await using tmp = await tmpdir()
    const { model } = await interleaved(tmp.path)

    expect(model.sessionID).toBe("")
    expect(model.systemPrompt).toBeNull()
    expect(model.toolExecutor).toBeNull()
    expect(model.sessionPreapprovedTools).toEqual([])
    expect(model.approvalHandler).toBeNull()
  })
})
