import { AsyncLocalStorage } from "node:async_hooks"
import type { FetchLike, Transport, TransportSendOptions } from "@modelcontextprotocol/sdk/shared/transport.js"
import type { JSONRPCMessage } from "@modelcontextprotocol/sdk/types.js"

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
      const response = await this.fetchImpl(url, init)
      if (response.status !== 429) return response
      if (!active && attempt >= MAX_UNSCOPED_RETRIES) return response

      const duration = delay(response.headers, attempt, { now: this.now(), random: this.random() })
      void response.body?.cancel().catch(() => {})
      await this.wait(duration, signals, active)
    }
  }
}

export function delay(headers: Headers, attempt: number, input: { now?: number; random?: number } = {}) {
  const now = input.now ?? Date.now()
  const retryAfter = parseRetryAfter(headers.get("retry-after"), now)
  if (retryAfter !== undefined && retryAfter > 0) return cap(retryAfter)

  const reset = parseSeconds(headers.get("ratelimit-reset"))
  if (reset !== undefined && reset > 0) return cap(reset * 1_000)

  const resetAfter = parseSeconds(headers.get("x-ratelimit-reset-after"))
  if (resetAfter !== undefined && resetAfter > 0) return cap(resetAfter * 1_000)

  const resetAt = parseSeconds(headers.get("x-ratelimit-reset"))
  if (resetAt !== undefined && resetAt * 1_000 > now) return cap(resetAt * 1_000 - now)

  const exponential = Math.min(INITIAL_DELAY * Math.pow(2, Math.max(0, Math.floor(attempt))), MAX_DELAY)
  const sample = input.random ?? Math.random()
  const jitter = Number.isFinite(sample) ? Math.min(1, Math.max(0, sample)) : 0.5
  return Math.ceil(exponential / 2 + (exponential / 2) * jitter)
}

function parseRetryAfter(value: string | null, now: number) {
  if (!value) return undefined
  const trimmed = value.trim()
  if (/^\d+$/.test(trimmed)) {
    const seconds = Number(trimmed)
    if (Number.isFinite(seconds)) return seconds * 1_000
    return undefined
  }
  // Numeric-looking non-integers are not valid Retry-After HTTP dates.
  if (/^[+-]?\d/.test(trimmed)) return undefined
  const date = Date.parse(trimmed)
  if (Number.isNaN(date)) return undefined
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
