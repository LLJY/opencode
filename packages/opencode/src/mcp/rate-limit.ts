import { AsyncLocalStorage } from "node:async_hooks"
import type { FetchLike, Transport, TransportSendOptions } from "@modelcontextprotocol/sdk/shared/transport.js"
import type { JSONRPCMessage } from "@modelcontextprotocol/sdk/types.js"
import { HttpDate } from "@/util/http-date"

const INITIAL_DELAY = 1_000
const MAX_DELAY = 30_000
const MAX_TIMER_DELAY = 2_147_483_647
const ACTIVE_POLL_INTERVAL = 25
const MAX_UNSCOPED_RETRIES = 3

type RemoteTransport = Transport & {
  finishAuth: (authorizationCode: string) => Promise<void>
  onsessionexpired?: () => Promise<void>
  terminateSession?: () => Promise<void>
  resumeStream?: (lastEventId: string, options?: { onresumptiontoken?: (token: string) => void }) => Promise<void>
  readonly protocolVersion?: string
}
type SendOptions = TransportSendOptions & { isRequestActive?: () => boolean }

export type Wait = (delay: number, signals: readonly AbortSignal[], active?: () => boolean) => Promise<void>

export interface Options {
  fetch?: FetchLike
  now?: () => number
  random?: () => number
  wait?: Wait
}

/**
 * Wraps an SDK HTTP transport so only definite HTTP 429 responses are replayed.
 * The SDK's request-active callback keeps retry waits inside its existing timeout.
 */
export class RetryTransport implements Transport {
  private readonly endpoint: URL
  private readonly transport: RemoteTransport
  private readonly requests = new AsyncLocalStorage<(() => boolean) | undefined>()
  private readonly controller = new AbortController()
  private readonly fetchImpl: FetchLike
  private readonly now: () => number
  private readonly random: () => number
  private readonly wait: Wait

  onclose?: () => void
  onerror?: (error: Error) => void
  onmessage?: Transport["onmessage"]

  constructor(endpoint: URL, make: (fetch: FetchLike) => RemoteTransport, options: Options = {}) {
    this.endpoint = endpoint
    this.fetchImpl = options.fetch ?? fetch
    this.now = options.now ?? Date.now
    this.random = options.random ?? Math.random
    this.wait = options.wait ?? wait
    this.transport = make((url, init) => this.retry(url, init))
    this.transport.onclose = () => {
      this.controller.abort()
      this.onclose?.()
    }
    this.transport.onerror = (error) => this.onerror?.(error)
    this.transport.onmessage = (message, extra) => this.onmessage?.(message, extra)
  }

  start() {
    return this.transport.start()
  }

  send(message: JSONRPCMessage, options?: SendOptions) {
    return this.requests.run(options?.isRequestActive, () => this.transport.send(message, options))
  }

  close() {
    this.controller.abort()
    return this.transport.close()
  }

  finishAuth(authorizationCode: string) {
    return this.transport.finishAuth(authorizationCode)
  }

  setProtocolVersion(version: string) {
    this.transport.setProtocolVersion?.(version)
  }

  get sessionId() {
    return this.transport.sessionId
  }

  get protocolVersion() {
    return this.transport.protocolVersion
  }

  terminateSession() {
    return this.transport.terminateSession?.() ?? Promise.resolve()
  }

  resumeStream(lastEventId: string, options?: { onresumptiontoken?: (token: string) => void }) {
    return this.transport.resumeStream?.(lastEventId, options) ?? Promise.resolve()
  }

  get onsessionexpired() {
    return this.transport.onsessionexpired
  }

  set onsessionexpired(value: (() => Promise<void>) | undefined) {
    this.transport.onsessionexpired = value
  }

  private async retry(url: string | URL, init?: RequestInit) {
    if (!isMcpRequest(url, init, this.endpoint)) return this.fetchImpl(url, init)
    const active = this.requests.getStore()
    const signals = init?.signal ? [this.controller.signal, init.signal] : [this.controller.signal]

    for (let attempt = 0; ; attempt++) {
      ensureActive(signals, active)

      // A hung fetch has to observe transport close, the caller's abort, and the SDK
      // request-active callback, so the attempt runs on a composed signal rather than
      // the caller's own. That signal outlives this call, so a returned SSE body still
      // aborts on transport close.
      // A rejection never says whether the server saw the request, so the abort — like
      // any other transport failure — is surfaced instead of replayed.
      const cancelled = new AbortController()
      const composed = AbortSignal.any([...signals, cancelled.signal])
      const poll = active
        ? setInterval(() => {
            if (active()) return
            cancelled.abort(abortError("MCP request was cancelled"))
          }, ACTIVE_POLL_INTERVAL)
        : undefined
      // A caller-supplied FetchLike may throw before it ever returns a promise, so
      // the poll is cleared from a finally rather than off a promise that in that
      // case was never created.
      let adopted = false
      try {
        const response = await this.fetchImpl(url, { ...init, signal: composed })

        const body = response.body
        if (response.status !== 429) {
          if (poll === undefined || body === null) return boundDiagnosticBody(response)
          adopted = true
          return boundDiagnosticBody(watchBody(response, body, composed, () => clearInterval(poll)))
        }
        // Replay stays keyed to the status read off the wire above, so giving up still
        // hands back a definite 429 — only its diagnostic body is capped.
        if (!active && attempt >= MAX_UNSCOPED_RETRIES) return boundDiagnosticBody(response)

        const duration = delay(response.headers, attempt, { now: this.now(), random: this.random() })
        void response.body?.cancel().catch(() => {})
        await this.wait(duration, signals, active)
      } finally {
        if (!adopted && poll !== undefined) clearInterval(poll)
      }
    }
  }
}

// Headers are not the end of an MCP request: a stalled JSON reply or a POST SSE
// stream is still the same in-flight request, so cancellation stays armed until that
// body completes, errors, or is cancelled. The signal watched here is the same one
// the fetch runs on, so transport close and the caller's own abort still tear the
// body down; this only extends them, and the SDK's request-active callback, past the
// headers. Settling on either end also bounds the poll's lifetime.
function watchBody(response: Response, source: NonNullable<Response["body"]>, signal: AbortSignal, stop: () => void) {
  const reader = source.getReader()
  let settled = false
  let abort: (() => void) | undefined
  const settle = () => {
    if (settled) return false
    settled = true
    if (abort) signal.removeEventListener("abort", abort)
    stop()
    return true
  }

  const body = new ReadableStream({
    start(controller) {
      abort = () => {
        if (!settle()) return
        controller.error(abortError(signal.reason))
        void reader.cancel(signal.reason).catch(() => {})
      }
      signal.addEventListener("abort", abort, { once: true })
      if (signal.aborted) abort()
    },
    async pull(controller) {
      const chunk = await reader.read().catch((error: unknown) => {
        if (settle()) controller.error(error)
        return undefined
      })
      if (chunk === undefined || settled) return
      if (!chunk.done) return controller.enqueue(chunk.value)
      settle()
      controller.close()
    },
    // The SDK awaits `response.body.cancel()` on its success paths — Streamable HTTP
    // 202 and the legacy SSE POST reply both release the connection that way — so
    // returning the source cancel would hand the whole transport to a server whose
    // cancel never settles. Releasing this wrapper is unconditional; the source cancel
    // is the server's promise to keep and is left to settle on its own.
    cancel(reason) {
      settle()
      void reader.cancel(reason).catch(() => {})
    },
  })

  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  })
}

// The SDK renders a failed request by awaiting `response.text()`, which reads the body
// to its end: a server answering an error with an endless or enormous stream would
// otherwise stall the transport or allocate without limit inside the SDK. Only an error
// body is capped — a success body is the transport's own message stream and has to keep
// flowing — and the status and headers pass through untouched, so nothing about which
// responses are replayed changes. Gating on 4xx/5xx rather than `!response.ok` also
// keeps the wrapper away from the statuses that forbid a body; the SDK never asks for a
// redirect it has to read itself.
const MAX_DIAGNOSTIC_BODY_BYTES = 16_384

function boundDiagnosticBody(response: Response) {
  const source = response.status < 400 ? null : response.body
  if (source === null) return response

  const reader = source.getReader()
  let remaining = MAX_DIAGNOSTIC_BODY_BYTES
  const body = new ReadableStream({
    async pull(controller) {
      const chunk = await reader.read().catch((error: unknown) => {
        controller.error(error)
        return undefined
      })
      if (chunk === undefined) return
      if (chunk.done) return controller.close()
      // Clipped rather than dropped: the budget is spent on the bytes that fit, so one
      // enormous frame costs the budget instead of its own length.
      const slice = chunk.value.length <= remaining ? chunk.value : chunk.value.slice(0, remaining)
      remaining -= slice.length
      controller.enqueue(slice)
      if (remaining > 0) return
      controller.close()
      void reader.cancel().catch(() => {})
    },
    cancel(reason) {
      void reader.cancel(reason).catch(() => {})
    },
  })

  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  })
}

export function delay(headers: Headers, attempt: number, input: { now?: number; random?: number } = {}) {
  const now = input.now ?? Date.now()
  const hint = headerDelay(headers, now)
  if (hint !== undefined && hint > 0) return cap(hint)

  const exponential = Math.min(INITIAL_DELAY * Math.pow(2, Math.max(0, Math.floor(attempt))), MAX_DELAY)
  const sample = input.random ?? Math.random()
  const jitter = Number.isFinite(sample) ? Math.min(1, Math.max(0, sample)) : 0.5
  return Math.ceil(exponential / 2 + (exponential / 2) * jitter)
}

// The first header that parses wins outright. A server that sends `Retry-After: 0`
// or an already elapsed deadline is saying "no wait is required", which is not an
// invitation to honour a header it ranked lower, so a non-positive hint falls to
// bounded backoff instead of to the reset headers behind it. Only a header that
// fails its grammar defers to the next one.
function headerDelay(headers: Headers, now: number) {
  const retryAfter = parseRetryAfter(headers.get("retry-after"), now)
  if (retryAfter !== undefined) return retryAfter

  const reset = parseSeconds(headers.get("ratelimit-reset"))
  if (reset !== undefined) return reset * 1_000

  const resetAfter = parseSeconds(headers.get("x-ratelimit-reset-after"))
  if (resetAfter !== undefined) return resetAfter * 1_000

  const resetAt = parseSeconds(headers.get("x-ratelimit-reset"))
  if (resetAt !== undefined) return Math.max(0, resetAt * 1_000 - now)

  return undefined
}

// RFC 9110 5.6.7 delta-seconds is `1*DIGIT`, so a fractional or signed value is
// malformed and must fall through to the HTTP-date form rather than become a delay.
function parseRetryAfter(value: string | null, now: number) {
  if (!value) return undefined
  const trimmed = value.trim()
  if (/^\d+$/.test(trimmed)) {
    const seconds = Number(trimmed)
    if (Number.isFinite(seconds)) return seconds * 1_000
    return undefined
  }
  const date = HttpDate.parse(trimmed, now)
  if (date === undefined) return undefined
  return Math.max(0, date - now)
}

function parseSeconds(value: string | null) {
  if (!value || !/^\d+(?:\.\d+)?$/.test(value.trim())) return undefined
  const seconds = Number(value)
  if (!Number.isFinite(seconds)) return undefined
  return seconds
}

function cap(delay: number) {
  return Math.min(MAX_TIMER_DELAY, Math.max(0, Math.ceil(delay)))
}

function isMcpRequest(url: string | URL, init: RequestInit | undefined, endpoint: URL) {
  const method = init?.method?.toUpperCase() ?? "GET"
  if (method === "GET") {
    if (new URL(url).href !== endpoint.href) return false
    return new Headers(init?.headers).get("accept")?.includes("text/event-stream") === true
  }
  if (method === "DELETE") return new URL(url).href === endpoint.href
  if (method !== "POST" || typeof init?.body !== "string") return false

  const value = parseJson(init.body)
  const messages = Array.isArray(value) ? value : [value]
  return messages.length > 0 && messages.every(isRetryableMessage)
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value)
  } catch {
    return undefined
  }
}

function isRetryableMessage(value: unknown) {
  if (typeof value !== "object" || value === null || !("jsonrpc" in value) || value.jsonrpc !== "2.0") return false
  if ("method" in value) return typeof value.method === "string"
  return "id" in value && ("result" in value || "error" in value)
}

function ensureActive(signals: readonly AbortSignal[], active?: () => boolean) {
  const aborted = signals.find((signal) => signal.aborted)
  if (aborted) throw abortError(aborted.reason)
  if (active?.() === false) throw abortError("MCP request was cancelled")
}

function wait(delay: number, signals: readonly AbortSignal[], active?: () => boolean) {
  ensureActive(signals, active)
  if (delay === 0) return Promise.resolve()

  return new Promise<void>((resolve, reject) => {
    const signal = AbortSignal.any([...signals])
    const timeout = setTimeout(() => finish(resolve), delay)
    const interval = active
      ? setInterval(
          () => {
            if (active()) return
            finish(() => reject(abortError("MCP request was cancelled")))
          },
          Math.min(ACTIVE_POLL_INTERVAL, delay),
        )
      : undefined
    const abort = () => finish(() => reject(abortError(signal.reason)))
    const finish = (done: () => void) => {
      clearTimeout(timeout)
      if (interval) clearInterval(interval)
      signal.removeEventListener("abort", abort)
      done()
    }
    signal.addEventListener("abort", abort, { once: true })
    if (signal.aborted) abort()
  })
}

function abortError(reason: unknown) {
  if (reason instanceof Error) return reason
  return new DOMException(typeof reason === "string" ? reason : "This operation was aborted", "AbortError")
}

export * as McpRateLimit from "./rate-limit"
