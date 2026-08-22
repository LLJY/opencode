import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { llmClient } from "@opencode-ai/core/effect/app-node-platform"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import { Provider } from "@/provider/provider"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { serviceUse } from "@opencode-ai/core/effect/service-use"
import { Context, Effect, Layer } from "effect"
import * as Stream from "effect/Stream"
import { streamText, wrapLanguageModel, type ModelMessage, type Tool } from "ai"
import type { LLMEvent } from "@opencode-ai/llm"
import { LLMClient } from "@opencode-ai/llm/route"
import type { LLMClientService } from "@opencode-ai/llm/route"
import { GitLabWorkflowLanguageModel } from "gitlab-ai-provider"
import { ProviderTransform } from "@/provider/transform"
import { Config } from "@/config/config"
import type { Agent } from "@/agent/agent"
import type { MessageV2 } from "./message-v2"
import { Plugin } from "@/plugin"
import { Permission } from "@/permission"
import { SessionID } from "@/session/schema"
import { Auth } from "@/auth"
import { EffectBridge } from "@/effect/bridge"
import { RuntimeFlags } from "@/effect/runtime-flags"
import * as Option from "effect/Option"
import * as OtelTracer from "@effect/opentelemetry/Tracer"
import { LLMAISDK } from "./llm/ai-sdk"
import { LLMNativeRuntime } from "./llm/native-runtime"
import { LLMRequestPrep } from "./llm/request"

export const OUTPUT_TOKEN_MAX = ProviderTransform.OUTPUT_TOKEN_MAX

export type StreamInput = {
  user: SessionV1.User
  sessionID: string
  parentSessionID?: string
  model: Provider.Model
  agent: Agent.Info
  permission?: PermissionV1.Ruleset
  system: string[]
  messages: ModelMessage[]
  small?: boolean
  tools: Record<string, Tool>
  retries?: number
  toolChoice?: "auto" | "required" | "none"
  previousResponseId?: string
}

export type StreamRequest = StreamInput & {
  abort: AbortSignal
}

export interface Interface {
  readonly stream: (input: StreamInput) => Stream.Stream<LLMEvent, unknown>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/LLM") {}

export const use = serviceUse(Service)

const live: Layer.Layer<
  Service,
  never,
  | Auth.Service
  | Config.Service
  | Provider.Service
  | Plugin.Service
  | Permission.Service
  | LLMClientService
  | RuntimeFlags.Service
> = Layer.effect(
  Service,
  Effect.gen(function* () {
    const auth = yield* Auth.Service
    const config = yield* Config.Service
    const provider = yield* Provider.Service
    const plugin = yield* Plugin.Service
    const perm = yield* Permission.Service
    const llmClient = yield* LLMClient.Service
    const flags = yield* RuntimeFlags.Service

    const run = Effect.fn("LLM.run")(function* (input: StreamRequest) {
      yield* Effect.logInfo("stream", {
        providerID: input.model.providerID,
        modelID: input.model.id,
        "session.id": input.sessionID,
        small: (input.small ?? false).toString(),
        agent: input.agent.name,
        mode: input.agent.mode,
      })

      const [language, cfg, item, info] = yield* Effect.all(
        [
          provider.getLanguage(input.model),
          config.get(),
          provider.getProvider(input.model.providerID),
          auth.get(input.model.providerID),
        ],
        { concurrency: "unbounded" },
      )

      const isWorkflow = language instanceof GitLabWorkflowLanguageModel
      const prepared = yield* LLMRequestPrep.prepare({
        ...input,
        provider: item,
        auth: info,
        plugin,
        flags,
        isWorkflow,
      })

      const workflowModel = isWorkflow ? createWorkflowModelFacade(language) : undefined

      // Wire up toolExecutor for DWS workflow models so that tool calls
      // from the workflow service are executed via opencode's tool system
      // and results sent back over the WebSocket.
      const bridge = yield* EffectBridge.make()
      if (workflowModel) {
        workflowModel.sessionID = input.sessionID
        workflowModel.systemPrompt = prepared.system.join("\n")
        workflowModel.toolExecutor = async (toolName, argsJson, _requestID) => {
          const t = prepared.tools[toolName]
          if (!t || !t.execute) {
            return { result: "", error: `Unknown tool: ${toolName}` }
          }
          try {
            const result = await t.execute!(JSON.parse(argsJson), {
              toolCallId: _requestID,
              messages: input.messages,
              abortSignal: input.abort,
            })
            const output = typeof result === "string" ? result : (result?.output ?? JSON.stringify(result))
            return {
              result: output,
              metadata: typeof result === "object" ? result?.metadata : undefined,
              title: typeof result === "object" ? result?.title : undefined,
            }
          } catch (e: any) {
            return { result: "", error: e.message ?? String(e) }
          }
        }

        workflowModel.sessionPreapprovedTools = workflowPreapprovedTools(
          Object.keys(prepared.tools),
          Permission.merge(input.agent.permission ?? [], input.permission ?? []),
        )

        const approvedToolsForSession = new Set<string>()
        workflowModel.approvalHandler = bridge.bind(async (approvalTools) => {
          const uniqueNames = [...new Set(approvalTools.map((t: { name: string }) => t.name))] as string[]
          // Auto-approve tools that were already approved in this session
          // (prevents infinite approval loops for server-side MCP tools)
          if (uniqueNames.every((name) => approvedToolsForSession.has(name))) {
            return { approved: true }
          }

          try {
            const approval = await bridge.promise(
              waitForWorkflowToolApproval({
                permission: perm,
                abort: input.abort,
                sessionID: input.sessionID,
                tools: approvalTools,
              }),
            )
            if (!approval.approved) return approval
            for (const name of uniqueNames) approvedToolsForSession.add(name)
            workflowModel.sessionPreapprovedTools = [...(workflowModel.sessionPreapprovedTools ?? []), ...uniqueNames]
            return approval
          } catch {
            return { approved: false }
          }
        })
      }

      const tracer = cfg.experimental?.openTelemetry
        ? Option.getOrUndefined(yield* Effect.serviceOption(OtelTracer.OtelTracer))
        : undefined
      const telemetryTracer = tracer
        ? new Proxy(tracer, {
            get(target, prop, receiver) {
              if (prop !== "startSpan") return Reflect.get(target, prop, receiver)
              return (...args: Parameters<typeof target.startSpan>) => {
                const span = target.startSpan(...args)
                span.setAttribute("session.id", input.sessionID)
                return span
              }
            },
          })
        : undefined

      // Runtime seam: native is an opt-in adapter over @opencode-ai/llm. It
      // either returns a ready LLMEvent stream or a concrete fallback reason.
      if (flags.experimentalNativeLlm) {
        const native = LLMNativeRuntime.stream({
          model: input.model,
          provider: item,
          auth: info,
          llmClient,
          messages: prepared.messages,
          tools: prepared.tools,
          toolChoice: input.toolChoice,
          temperature: prepared.params.temperature,
          topP: prepared.params.topP,
          topK: prepared.params.topK,
          maxOutputTokens: prepared.params.maxOutputTokens,
          providerOptions: prepared.params.options,
          headers: prepared.headers,
          abort: input.abort,
        })
        if (native.type === "supported") {
          yield* Effect.logInfo("llm runtime selected", {
            "llm.runtime": "native",
            "llm.provider": input.model.providerID,
            "llm.model": input.model.id,
          })
          return {
            type: "native" as const,
            stream: native.stream,
          }
        }
        yield* Effect.logInfo("llm runtime selected", {
          "llm.runtime": "ai-sdk",
          "llm.provider": input.model.providerID,
          "llm.model": input.model.id,
          "llm.native_unsupported_reason": native.reason,
        })
        yield* Effect.logInfo("native runtime unavailable; falling back to ai-sdk", {
          providerID: input.model.providerID,
          modelID: input.model.id,
          "session.id": input.sessionID,
          small: (input.small ?? false).toString(),
          agent: input.agent.name,
          mode: input.agent.mode,
          reason: native.reason,
        })
      }

      yield* Effect.logInfo("llm runtime selected", {
        "llm.runtime": "ai-sdk",
        "llm.provider": input.model.providerID,
        "llm.model": input.model.id,
      })
      // Default runtime path: AI SDK owns provider execution and tool dispatch;
      // LLMAISDK.toLLMEvents below normalizes fullStream parts for the processor.
      return {
        type: "ai-sdk" as const,
        result: streamText({
          onError(error) {
            bridge.fork(
              Effect.logError("stream error", {
                providerID: input.model.providerID,
                modelID: input.model.id,
                "session.id": input.sessionID,
                small: (input.small ?? false).toString(),
                agent: input.agent.name,
                mode: input.agent.mode,
                error,
              }),
            )
          },
          // Copilot returns the authoritative billed amount only in provider-specific response fields.
          includeRawChunks: input.model.providerID.includes("github-copilot"),
          async experimental_repairToolCall(failed) {
            const lower = failed.toolCall.toolName.toLowerCase()
            if (lower !== failed.toolCall.toolName && prepared.tools[lower]) {
              return {
                ...failed.toolCall,
                toolName: lower,
              }
            }
            return {
              ...failed.toolCall,
              input: JSON.stringify({
                tool: failed.toolCall.toolName,
                error: failed.error.message,
              }),
              toolName: "invalid",
            }
          },
          temperature: prepared.params.temperature,
          topP: prepared.params.topP,
          topK: prepared.params.topK,
          providerOptions: ProviderTransform.providerOptions(input.model, prepared.params.options),
          activeTools: Object.keys(prepared.tools).filter((x) => x !== "invalid"),
          tools: prepared.tools,
          toolChoice: input.toolChoice,
          maxOutputTokens: prepared.params.maxOutputTokens,
          abortSignal: input.abort,
          headers: prepared.headers,
          maxRetries: input.retries ?? 0,
          messages: prepared.messages,
          model: wrapLanguageModel({
            model: workflowModel ?? language,
            middleware: [
              {
                specificationVersion: "v3" as const,
                async transformParams(args) {
                  if (args.type === "stream") {
                    // @ts-expect-error
                    args.params.prompt = ProviderTransform.message(
                      args.params.prompt,
                      input.model,
                      prepared.messageTransformOptions,
                    )
                  }
                  return args.params
                },
              },
            ],
          }),
          experimental_telemetry: {
            isEnabled: cfg.experimental?.openTelemetry,
            functionId: "session.llm",
            tracer: telemetryTracer,
            metadata: {
              userId: cfg.username ?? "unknown",
              sessionId: input.sessionID,
            },
          },
        }),
      }
    })

    const stream: Interface["stream"] = (input) =>
      Stream.scoped(
        Stream.unwrap(
          Effect.gen(function* () {
            const ctrl = yield* Effect.acquireRelease(
              Effect.sync(() => new AbortController()),
              (ctrl) => Effect.sync(() => ctrl.abort()),
            )

            const result = yield* run({ ...input, abort: ctrl.signal })

            if (result.type === "native") return result.stream

            // Adapter seam: both runtimes expose the same LLMEvent stream. Native
            // already returns one; AI SDK streams are converted here.
            const state = LLMAISDK.adapterState()
            return Stream.fromAsyncIterable(result.result.fullStream, (e) =>
              e instanceof Error ? e : new Error(String(e)),
            ).pipe(
              Stream.mapEffect((event) => LLMAISDK.toLLMEvents(state, event)),
              Stream.flatMap((events) => Stream.fromIterable(events)),
            )
          }),
        ),
      )

    return Service.of({ stream })
  }),
)

// One vendor model instance backs every session: `Provider.getLanguage` memoizes it
// per provider/model, and the model itself is built to be shared — it keys DWS
// workflows by session in `sessionWorkflows` and tracks every live client in
// `activeClients`. Only the values the host rewrites per stream are the exception, so
// they live on this facade rather than on the shared instance. `doStream` snapshots
// `sessionID`, `toolExecutor`, and `sessionPreapprovedTools` at entry and the approval
// path reads `workflowOptions.approvalHandler` live, so two interleaved sessions would
// otherwise run each other's tools, prompt, and permission requests.
//
// Both traps pass the facade as the receiver. That is what makes the vendor's
// prototype accessors resolve against request state — `toolExecutor` writes
// `this._toolExecutor`, `systemPrompt` and `approvalHandler` write through
// `this.workflowOptions` — while an ordinary cache write such as
// `this.detectedProjectPath` still lands on the shared instance.
//
// Only `[[Get]]` and `[[Set]]` are trapped, so a descriptor read still reports the
// shared instance's value. Nothing in the vendor, `wrapLanguageModel`, or `streamText`
// reads descriptors; anything that does — `Object.freeze`, `structuredClone` — needs a
// trap of its own.
export function createWorkflowModelFacade<T extends GitLabWorkflowLanguageModel>(model: T): T {
  // `workflowOptions` is `private` in the vendor types, but it is the object both the
  // `systemPrompt` and `approvalHandler` setters write through, so the facade needs
  // its own copy. `onUsageUpdate` and `onSelectModel` are documented per-stream host
  // callbacks that nothing sets yet; they are listed so wiring one up later cannot
  // quietly reintroduce the sharing this facade exists to prevent. The rest are seeded
  // with the vendor's own defaults so a facade starts where a fresh instance would.
  const shared = model as unknown as { readonly workflowOptions: Record<string, unknown> }
  const request = new Map<PropertyKey, unknown>([
    ["sessionID", ""],
    ["sessionPreapprovedTools", []],
    ["_toolExecutor", null],
    ["onUsageUpdate", null],
    ["onSelectModel", null],
    ["workflowOptions", { ...shared.workflowOptions }],
  ])

  return new Proxy(model, {
    get: (target, property, facade) =>
      request.has(property) ? request.get(property) : Reflect.get(target, property, facade),
    set: (target, property, value, facade) => {
      if (!request.has(property)) return Reflect.set(target, property, value, facade)
      request.set(property, value)
      return true
    },
  })
}

// The workflow service treats this list as "already authorized" and skips
// `approvalHandler` for every name on it, so it is the only gate in front of the tool
// executor — and that executor runs the local handler, plugin `tool.execute.before`
// hook included. A preapproval is decided before any arguments exist, so it only holds
// if the rule holds for every pattern: the local evaluator is asked with the tool's
// permission alias and the `*` pattern, and only an explicit `allow` counts. An
// unmatched tool falls through the evaluator's `ask` default, so `ask` and `deny` alike
// stay off the list and reach the workflow's approval round trip first.
export function workflowPreapprovedTools(tools: ReadonlyArray<string>, ruleset: PermissionV1.Ruleset) {
  return tools.filter((name) => Permission.evaluate(Permission.alias(name), "*", ruleset).action === "allow")
}

// `Permission.ask` removes a pending request when interrupted. Keeping the abort race
// in the same Effect ensures that finalizer runs before the workflow sees a rejection.
// The permission service is passed in rather than yielded from context. This runs
// through `EffectBridge.promise`, which erases requirements with a cast, so an ambient
// service would resolve only as long as the captured context happened to carry it —
// and getting that wrong would surface at runtime instead of failing to compile.
export function waitForWorkflowToolApproval(input: {
  permission: Permission.Interface
  abort: AbortSignal
  sessionID: string
  tools: ReadonlyArray<{ name: string; args: string }>
}) {
  return Effect.gen(function* () {
    const patterns = [
      ...new Set(
        input.tools.map((tool) => {
          try {
            const parsed = JSON.parse(tool.args) as Record<string, unknown>
            const title = (parsed?.title ?? parsed?.name ?? "") as string
            return title ? `${tool.name}: ${title}` : tool.name
          } catch {
            return tool.name
          }
        }),
      ),
    ]

    return yield* Effect.raceFirst(
      input.permission
        .ask({
          id: PermissionV1.ID.ascending(),
          sessionID: SessionID.make(input.sessionID),
          permission: "workflow_tool_approval",
          patterns,
          metadata: { tools: input.tools },
          always: patterns,
          ruleset: [],
        })
        .pipe(Effect.as({ approved: true as const })),
      waitForAbort(input.abort).pipe(Effect.as({ approved: false as const })),
    )
  })
}

function waitForAbort(signal: AbortSignal) {
  return Effect.callback<void>((resume) => {
    if (signal.aborted) return resume(Effect.void)
    const onAbort = () => resume(Effect.void)
    signal.addEventListener("abort", onAbort, { once: true })
    return Effect.sync(() => signal.removeEventListener("abort", onAbort))
  })
}

export const hasToolCalls = LLMRequestPrep.hasToolCalls

export const node = LayerNode.make({
  service: Service,
  layer: live,
  deps: [Auth.node, Config.node, Provider.node, Plugin.node, Permission.node, llmClient, RuntimeFlags.node],
})

export * as LLM from "./llm"
