import { expect, test } from "bun:test"
import {
  AIError,
  AuthenticationError,
  ContentPolicyError,
  HttpContext,
  InvalidRequestError,
  RateLimitError,
  TransportError,
} from "@opencode/ai"
import { SessionRunnerRetry } from "@opencode/core/session/runner/retry"
import { Session } from "@opencode/core/session"
import { Agent } from "@opencode/core/agent"
import { Model } from "@opencode/core/model"
import { Provider } from "@opencode/core/provider"
import { Effect } from "effect"

for (const reason of [
  new AuthenticationError({ message: "Denied" }),
  new ContentPolicyError({ message: "Denied" }),
  new InvalidRequestError({ message: "Invalid" }),
]) {
  test(`${reason._tag} cannot be made retryable by a positive header or plugin`, async () => {
    const cause = new AIError({
      reason: Object.assign(reason, {
        http: HttpContext.make({ url: "https://example.test", status: 400, headers: { "x-should-retry": "true" } }),
      }),
    })
    expect(SessionRunnerRetry.isRetryable(cause)).toBe(false)
    const decide = await Effect.runPromise(SessionRunnerRetry.policy(Session.ID.create()))
    const result = await Effect.runPromise(
      decide({
        cause,
        error: { type: "provider.invalid-request", message: cause.message },
        agent: Agent.defaultID,
        model: { id: Model.ID.make("test"), providerID: Provider.ID.opencode },
        retry: false,
        hook: (event) =>
          Effect.sync(() => {
            event.decision = { retry: true, delay: 0 }
          }),
      }),
    )
    expect(result).toEqual({ retry: false })
  })
}

test("the upstream timeout retry cap cannot be overridden by a plugin", async () => {
  const decide = await Effect.runPromise(SessionRunnerRetry.policy(Session.ID.create()))
  const results = []
  for (const _ of [1, 2, 3, 4]) {
    results.push(
      await Effect.runPromise(
        decide({
          cause: new AIError({
            reason: new TransportError({ message: "Timeout", code: "Timeout", transport: "http", operation: "read" }),
          }),
          error: { type: "provider.transport", message: "Timeout" },
          agent: Agent.defaultID,
          model: { id: Model.ID.make("test"), providerID: Provider.ID.opencode },
          retry: true,
          hook: (event) =>
            Effect.sync(() => {
              event.decision = { retry: true, delay: 0 }
            }),
        }),
      ),
    )
  }
  expect(results.map((result) => result.retry)).toEqual([true, true, true, false])
})

test("an explicit negative retry decision cannot be overridden by a plugin", async () => {
  const decide = await Effect.runPromise(SessionRunnerRetry.policy(Session.ID.create()))
  expect(
    await Effect.runPromise(
      decide({
        cause: new AIError({ reason: new RateLimitError({ message: "Busy" }) }),
        error: { type: "provider.rate-limit", message: "Busy" },
        agent: Agent.defaultID,
        model: { id: Model.ID.make("test"), providerID: Provider.ID.opencode },
        retry: false,
        hook: (event) =>
          Effect.sync(() => {
            event.decision = { retry: true, delay: 0 }
          }),
      }),
    ),
  ).toEqual({ retry: false })
})
