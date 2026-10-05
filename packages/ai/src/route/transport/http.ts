import { Clock, Duration, Effect, Stream } from "effect"
import { Headers, HttpClientRequest } from "effect/unstable/http"
import { Auth } from "../auth.js"
import { render as renderEndpoint } from "../endpoint.js"
import { Framing } from "../framing.js"
import type { HttpMiddleware, Transport, TransportPrepareInput } from "./index.js"
import * as ProviderShared from "../../protocols/shared.js"
import {
  AIError,
  DEFAULT_HTTP_TIMEOUT_MS,
  mergeJsonRecords,
  TransportError,
  type HttpContext,
  type HttpTimeout,
  type LLMRequest,
} from "../../schema/index.js"
import { RequestExecutor } from "../executor.js"

export type JsonRequestInput<Body> = TransportPrepareInput<Body>

export interface JsonRequestParts<Body = unknown> {
  readonly url: string
  readonly jsonBody: Body | Record<string, unknown>
  readonly bodyText: string
  readonly headers: Headers.Headers
}

export interface HttpPrepared<Frame> {
  readonly request: HttpClientRequest.HttpClientRequest
  readonly framing: Framing.Definition<Frame>
  readonly middleware?: HttpMiddleware
}

const applyQuery = (url: string, query: Record<string, string> | undefined) => {
  if (!query) return url
  const next = new URL(url)
  Object.entries(query).forEach(([key, value]) => next.searchParams.set(key, value))
  return next.toString()
}

const bodyWithOverlay = <Body>(body: Body, request: LLMRequest, encodeBody: (body: Body) => string) =>
  Effect.gen(function* () {
    if (request.http?.body === undefined) return { jsonBody: body, bodyText: encodeBody(body) }
    if (ProviderShared.isRecord(body)) {
      const overlaid = mergeJsonRecords(body, request.http.body) ?? {}
      return { jsonBody: overlaid, bodyText: ProviderShared.encodeJson(overlaid) }
    }
    return yield* ProviderShared.invalidRequest("http.body can only overlay JSON object request bodies")
  })

export const jsonRequestParts = <Body>(input: JsonRequestInput<Body>) =>
  Effect.gen(function* () {
    const url = applyQuery(
      renderEndpoint(input.endpoint, { request: input.request, body: input.body }).toString(),
      input.request.http?.query,
    )
    const body = yield* bodyWithOverlay(input.body, input.request, input.encodeBody)
    const headers = yield* Auth.toEffect(input.auth)({
      request: input.request,
      method: "POST",
      url,
      body: body.bodyText,
      headers: Headers.fromInput({
        ...input.headers?.({ request: input.request }),
        ...input.request.http?.headers,
      }),
    })
    return { url, jsonBody: body.jsonBody, bodyText: body.bodyText, headers }
  })

export interface HttpJsonInput<_Body, Frame> {
  readonly framing: Framing.Definition<Frame>
}

export type HttpJsonPatch<Body, Frame> = Partial<HttpJsonInput<Body, Frame>>

export interface HttpJsonTransport<Body, Frame> extends Transport<Body, HttpPrepared<Frame>, Frame> {
  readonly with: (patch: HttpJsonPatch<Body, Frame>) => HttpJsonTransport<Body, Frame>
}

export const httpJson = <Body, Frame>(input: HttpJsonInput<Body, Frame>): HttpJsonTransport<Body, Frame> => ({
  id: "http-json",
  with: (patch) => httpJson({ ...input, ...patch }),
  prepare: (prepareInput) =>
    Effect.gen(function* () {
      const parts = yield* jsonRequestParts({ ...prepareInput })
      const request = ProviderShared.jsonPost({
        url: parts.url,
        body: parts.bodyText,
        headers: parts.headers,
      })
      return {
        request,
        framing: input.framing,
        middleware: prepareInput.middleware,
      }
    }),
  execute: (prepared, request, runtime) =>
    Effect.gen(function* () {
      const timeout = (operation: "request" | "read", message: string, http?: HttpContext) =>
        new AIError({
          reason: new TransportError({
            message,
            transport: "http",
            operation,
            code: "Timeout",
            url: prepared.request.url,
            http,
          }),
        })
      const started = yield* Clock.currentTimeMillis
      // Unlike the header and chunk limits, the whole-request budget has no default.
      const total = request.http?.timeout ? Duration.millis(request.http.timeout) : Duration.infinity
      const response = yield* runtime.http.execute(prepared.request, prepared.middleware).pipe(
        Effect.timeoutOrElse({
          duration: Duration.min(timeoutDuration(request.http?.headerTimeout), total),
          orElse: () => timeout("request", "Timed out waiting for response headers"),
        }),
      )
      const http = RequestExecutor.responseHttp(response)
      return {
        // Charge active upstream waits for framed progress, not comment heartbeats or consumer pauses.
        // Pulls can return many frames. Check queued frames too, without charging chunk wait for consumer pauses.
        frames: Stream.transformPull(prepared.framing.frame(RequestExecutor.responseStream(response)), (pull) =>
          Effect.succeed(
            Effect.gen(function* () {
              const remaining = Duration.subtract(total, Duration.millis((yield* Clock.currentTimeMillis) - started))
              if (Duration.toMillis(remaining) <= 0)
                return yield* timeout("read", "Timed out waiting for the response to complete", http)
              return yield* pull.pipe(
                Effect.timeoutOrElse({
                  duration: Duration.min(timeoutDuration(request.http?.chunkTimeout), remaining),
                  orElse: () => timeout("read", "Timed out waiting for response data", http),
                }),
              )
            }),
          ),
        ).pipe(
          Stream.mapEffect((frame) => checkDeadline(started + Duration.toMillis(total), http).pipe(Effect.as(frame))),
        ),
        http,
        body: prepared.framing.body,
      }
    }),
})

/** A frame or native event can remain buffered after the network pull has completed. */
export const checkDeadline = (deadline: number, http: HttpContext) =>
  deadline === Infinity
    ? Effect.void
    : Effect.gen(function* () {
        if ((yield* Clock.currentTimeMillis) < deadline) return
        return yield* new AIError({
          reason: new TransportError({
            message: "Timed out waiting for the response to complete",
            transport: "http",
            operation: "read",
            code: "Timeout",
            url: http.url,
            http,
          }),
        })
      })

const timeoutDuration = (value: HttpTimeout | undefined) =>
  value === false ? Duration.infinity : Duration.millis(value ?? DEFAULT_HTTP_TIMEOUT_MS)

export const sseJson = {
  id: "http-json/sse",
  with: <Body>() => httpJson<Body, string>({ framing: Framing.sse }),
} as const
