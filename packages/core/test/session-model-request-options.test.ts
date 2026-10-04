import { expect } from "bun:test"
import { HttpOptions, LanguageModel, mergeHttpOptions, Message } from "@opencode/ai"
import { LLMClient, RequestExecutor } from "@opencode/ai/route"
import { Agent } from "@opencode/schema/agent"
import { Money } from "@opencode/schema/money"
import { Session } from "@opencode/schema/session"
import { Location } from "@opencode/core/location"
import { Model } from "@opencode/core/model"
import { ModelResolver } from "@opencode/core/model-resolver"
import { PluginHooks } from "@opencode/core/plugin/hooks"
import { Project } from "@opencode/core/project"
import { Provider } from "@opencode/core/provider"
import { AbsolutePath } from "@opencode/core/schema"
import { SessionModelRequest } from "@opencode/core/session/model-request"
import { SessionModelTransport } from "@opencode/core/session/model-transport"
import { Cause, DateTime, Effect, Fiber, Layer } from "effect"
import { FetchHttpClient } from "effect/unstable/http"
import { testEffect } from "./lib/effect"
import { PluginTestLayer } from "./plugin/fixture"

const it = testEffect(PluginTestLayer)
const session = Session.Info.make({
  id: Session.ID.make("ses_configured_timeout"),
  projectID: Project.ID.global,
  cost: Money.USD.zero,
  tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
  time: { created: DateTime.makeUnsafe(0), updated: DateTime.makeUnsafe(0) },
  location: Location.Ref.make({ directory: AbsolutePath.make("/project") }),
})
const transport = SessionModelTransport.Service.of({
  bind: () => ({ execute: () => Effect.die("HTTP configuration must not force WebSocket") }),
  close: () => Effect.void,
  closeAll: Effect.void,
})
const runtime = LLMClient.layer.pipe(Layer.provide(RequestExecutor.layer.pipe(Layer.provide(FetchHttpClient.layer))))

for (const mode of ["request", "chunk", "disabled", "cancel"] as const)
  it.live(`configured native HTTP ${mode} timeout policy reaches the session request`, () =>
    Effect.gen(function* () {
      const timers: Array<ReturnType<typeof setTimeout>> = []
      let cancelled = false
      let received: Request | undefined
      const server = yield* Effect.acquireRelease(
        Effect.sync(() =>
          Bun.serve({
            hostname: "127.0.0.1",
            port: 0,
            fetch(request) {
              received = request
              return new Response(
                new ReadableStream({
                  start(controller) {
                    const encoder = new TextEncoder()
                    for (let i = 0; i < (mode === "chunk" ? 2 : 12); i++)
                      timers.push(
                        setTimeout(
                          () => {
                            if (cancelled) return
                            controller.enqueue(
                              encoder.encode(
                                `data: ${JSON.stringify({ id: "chunk", object: "chat.completion.chunk", created: 0, model: "test", choices: [{ index: 0, delta: { content: "x" }, finish_reason: i === (mode === "chunk" ? 1 : 11) ? "stop" : null }] })}\n\n${i === (mode === "chunk" ? 1 : 11) ? "data: [DONE]\n\n" : ""}`,
                              ),
                            )
                            if (i === (mode === "chunk" ? 1 : 11)) controller.close()
                          },
                          i * (mode === "chunk" ? 200 : 20),
                        ),
                      )
                  },
                  cancel() {
                    cancelled = true
                  },
                }),
                { headers: { "content-type": "text/event-stream" } },
              )
            },
          }),
        ),
        (server) =>
          Effect.sync(() => {
            cancelled = true
            timers.forEach(clearTimeout)
            server.stop(true)
          }),
      )
      const provider = Provider.Info.make({
        ...Provider.Info.empty(Provider.ID.make("timeout-fixture")),
        activation: "enabled",
        package: "@opencode/ai/providers/openai/chat",
        settings: {
          baseURL: server.url.href,
          ...(mode === "chunk" ? {} : { timeout: mode === "disabled" ? false : mode === "cancel" ? 500 : 70 }),
          chunkTimeout: mode === "chunk" ? 70 : mode === "cancel" ? 500 : 150,
        },
      })
      const selected = Model.Info.make({
        ...Model.Info.default(provider.id, Model.ID.make("test")),
        package: provider.package,
        headers: { "x-configured": "kept" },
        body: { custom_extension: { kept: true } },
        settings: { timeout: 1, chunkTimeout: 1, transport: "websocket" },
      })
      const resolver = yield* ModelResolver.Service.pipe(
        Effect.provide(ModelResolver.layer),
        Effect.provide(Layer.mock(Provider.Service, { get: () => Effect.succeed(provider) })),
      )
      const resolved = yield* resolver.resolveModel(selected)
      const hooks = yield* PluginHooks.Service
      yield* hooks.register("session", "model.request", (event) =>
        Effect.sync(() => {
          event.headers["x-hook"] = "kept"
        }),
      )
      const requests = yield* SessionModelRequest.Service.pipe(
        Effect.provide(SessionModelRequest.layer),
        Effect.provideService(SessionModelTransport.Service, transport),
      )
      const prepared = yield* requests.primary({
        session,
        agent: Agent.ID.make("build"),
        model: {
          ...resolved,
          // 022 resolves provider timeouts into model defaults; extend, do not replace, that policy.
          model: LanguageModel.update(resolved.model, {
            defaults: {
              ...resolved.model.defaults,
              http: mergeHttpOptions(resolved.model.defaults?.http, new HttpOptions({ query: { overlay: "kept" } })),
            },
          }),
        },
        system: [],
        messages: [Message.user("Hello")],
        webSocket: "session",
      })
      expect(prepared.request.http).toMatchObject({
        timeout: provider.settings?.timeout,
        chunkTimeout: provider.settings?.chunkTimeout,
        query: { overlay: "kept" },
        body: { custom_extension: { kept: true } },
        headers: { "x-configured": "kept", "x-hook": "kept", "x-opencode-session-id": session.id },
      })
      expect(prepared.options.webSocket).toBeUndefined()
      expect(prepared.request.providerOptions?.timeout).toBeUndefined()
      const generate = LLMClient.generate(prepared.request, prepared.options).pipe(Effect.provide(runtime))
      if (mode === "cancel") {
        const fiber = yield* Effect.forkChild(generate)
        yield* Effect.sleep(40)
        yield* Fiber.interrupt(fiber)
        const exit = yield* Fiber.await(fiber)
        expect(exit._tag).toBe("Failure")
        if (exit._tag === "Failure") expect(Cause.hasInterrupts(exit.cause)).toBe(true)
      } else {
        const result = yield* generate.pipe(Effect.result)
        expect(result._tag).toBe(mode === "disabled" ? "Success" : "Failure")
        if (result._tag === "Failure")
          expect(result.failure.reason).toMatchObject({
            _tag: "Transport",
            code: "Timeout",
            operation: "read",
            http: { status: 200 },
          })
      }
      expect(received?.headers.get("x-hook")).toBe("kept")
      expect(received?.url).toContain("overlay=kept")
    }),
  )
