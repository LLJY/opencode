import { Cause, Context, Deferred, Effect, Exit, Fiber, Layer, Random, Stream } from "effect"
import {
  FetchHttpClient,
  Headers,
  HttpClient,
  HttpClientError,
  HttpClientRequest,
  HttpClientResponse,
} from "effect/unstable/http"
import {
  AuthenticationReason,
  ContentPolicyReason,
  HttpContext,
  HttpRateLimitDetails,
  HttpRequestDetails,
  HttpResponseDetails,
  InvalidRequestReason,
  LLMError,
  ProviderInternalReason,
  QuotaExceededReason,
  RateLimitReason,
  TransportReason,
  UnknownProviderReason,
} from "../schema"
import { isContextOverflow } from "../provider-error"
import { parseHttpDate } from "../utils/http-date"

export interface Interface {
  readonly execute: (
    request: HttpClientRequest.HttpClientRequest,
  ) => Effect.Effect<HttpClientResponse.HttpClientResponse, LLMError>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/LLM/RequestExecutor") {}

// The diagnostic budget. It is enforced on the bytes off the wire before anything is
// decoded — a provider answering an error with megabytes of multi-byte text must not
// be able to spend a character budget on an unbounded allocation — and again on the
// redacted text that is finally reported.
const BODY_LIMIT = 16_384
// Redaction runs over the whole read before truncation, so the read deliberately
// overshoots the diagnostic budget: a secret straddling BODY_LIMIT is still whole
// in the buffer and gets redacted before the slice, exactly as it would have been
// with an unbounded read. The overshoot doubles as the sentinel that proves the
// provider sent more than the budget.
const BODY_READ_LIMIT = BODY_LIMIT + 4_096
const MAX_RETRIES = 2
const BASE_DELAY_MS = 500
const MAX_DELAY_MS = 10_000
const REDACTED = "<redacted>"

// One source of truth for what counts as a sensitive name across headers,
// URL query keys, and field names embedded inside request/response bodies.
//
// `SENSITIVE_NAME` is used as both a substring matcher (for free-form header
// names like `Authorization` / `X-API-Key`) and as the body-field alternation
// list. `SHORT_QUERY_NAME` covers anchored short keys like `?key=…` / `?sig=…`
// that are too generic to redact substring-style without false positives.
const SENSITIVE_NAME_SOURCE =
  "authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|token|secret|credential|signature|x-amz-signature"
const SENSITIVE_NAME = new RegExp(SENSITIVE_NAME_SOURCE, "i")
const SHORT_QUERY_NAME = /^(key|sig)$/i
const SENSITIVE_BODY_FIELD = new RegExp(`(?:${SENSITIVE_NAME_SOURCE}|key)`, "i")
// A JSON string value can contain an escaped quote, so the value pattern consumes
// escape pairs rather than stopping at the first `"` it sees. The escaped character is
// matched with `[\s\S]` rather than `.`, because `.` stops at a line terminator: an
// escaped newline would strand the value and report the whole thing verbatim.
const REDACT_JSON_FIELD = new RegExp(
  `("(?:${SENSITIVE_BODY_FIELD.source})"\\s*:\\s*)"(?:[^"\\\\]|\\\\[\\s\\S])*"`,
  "gi",
)
// A value with no closing quote is invisible to the terminated pass, so it would
// otherwise be reported verbatim — whether the bounded read cut the quote off or the
// provider never sent one. The pattern can only match a value that runs to the end of
// the text, so it never sees a complete field the terminated pass already replaced.
//
// The trailing `\\?` is what covers a cut that lands directly after a backslash: the
// escape pair has nothing left to consume, so without it the whole value falls out of
// the match and is reported in full.
const REDACT_UNTERMINATED_JSON_FIELD = new RegExp(
  `("(?:${SENSITIVE_BODY_FIELD.source})"\\s*:\\s*)"(?:[^"\\\\]|\\\\[\\s\\S])*\\\\?$`,
  "gi",
)
const REDACT_QUERY_FIELD = new RegExp(`((?:${SENSITIVE_BODY_FIELD.source})=)[^&\\s"]+`, "gi")

const isSensitiveHeaderName = (name: string) => SENSITIVE_NAME.test(name)

const isSensitiveQueryName = (name: string) => isSensitiveHeaderName(name) || SHORT_QUERY_NAME.test(name)

const redactHeaders = (headers: Headers.Headers, redactedNames: ReadonlyArray<string | RegExp>) =>
  Object.fromEntries(
    Object.entries(Headers.redact(headers, [...redactedNames, SENSITIVE_NAME])).map(([name, value]) => [
      name,
      String(value),
    ]),
  )

const redactUrl = (value: string) => {
  if (!URL.canParse(value)) return REDACTED
  const url = new URL(value)
  url.searchParams.forEach((_, key) => {
    if (isSensitiveQueryName(key)) url.searchParams.set(key, REDACTED)
  })
  return url.toString()
}

const normalizedHeaders = (headers: Headers.Headers) =>
  Object.fromEntries(Object.entries(headers).map(([key, value]) => [key.toLowerCase(), value]))

const requestId = (headers: Record<string, string>) => {
  return (
    headers["x-request-id"] ??
    headers["request-id"] ??
    headers["x-amzn-requestid"] ??
    headers["x-amz-request-id"] ??
    headers["x-goog-request-id"] ??
    headers["cf-ray"]
  )
}

const retryableStatus = (status: number) => status === 429 || status === 503 || status === 504 || status === 529

// `Number` is far looser than the header grammars: it accepts signs, exponents,
// hex, `Infinity`, and turns an empty value into 0, which would silently disable
// backoff. Only an unsigned number is a retry hint. Zero stays valid because
// providers use it to mean "retry immediately".
//
// RFC 9110 5.6.7 delta-seconds is `1*DIGIT`, so a standard `Retry-After: 1.5` is
// malformed. The proprietary `retry-after-ms` header has no such grammar and
// providers do send fractions there, so it keeps accepting them.
const DECIMAL = /^\d+(?:\.\d+)?$/
const DELTA_SECONDS = /^\d+$/
const unsigned = (value: string | undefined, grammar: RegExp) => {
  if (value === undefined) return undefined
  const trimmed = value.trim()
  if (!grammar.test(trimmed)) return undefined
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : undefined
}

const retryAfterMs = (headers: Record<string, string>) => {
  // A malformed higher-precedence header is not a hint, so fall through to
  // Retry-After rather than treating it as zero delay.
  const millis = unsigned(headers["retry-after-ms"], DECIMAL)
  if (millis !== undefined) return millis

  const value = headers["retry-after"]
  if (!value) return undefined

  const seconds = unsigned(value, DELTA_SECONDS)
  if (seconds !== undefined) return seconds * 1000

  const now = Date.now()
  const date = parseHttpDate(value.trim(), now)
  if (date === undefined) return undefined
  return Math.max(0, date - now)
}

const addRateLimitValue = (target: Record<string, string>, key: string, value: string) => {
  if (key.length > 0) target[key] = value
}

const rateLimitDetails = (headers: Record<string, string>, retryAfter: number | undefined) => {
  const limit: Record<string, string> = {}
  const remaining: Record<string, string> = {}
  const reset: Record<string, string> = {}

  Object.entries(headers).forEach(([name, value]) => {
    const openaiLimit = /^x-ratelimit-limit-(.+)$/.exec(name)?.[1]
    if (openaiLimit) return addRateLimitValue(limit, openaiLimit, value)

    const openaiRemaining = /^x-ratelimit-remaining-(.+)$/.exec(name)?.[1]
    if (openaiRemaining) return addRateLimitValue(remaining, openaiRemaining, value)

    const openaiReset = /^x-ratelimit-reset-(.+)$/.exec(name)?.[1]
    if (openaiReset) return addRateLimitValue(reset, openaiReset, value)

    const anthropic = /^anthropic-ratelimit-(.+)-(limit|remaining|reset)$/.exec(name)
    if (!anthropic) return
    if (anthropic[2] === "limit") return addRateLimitValue(limit, anthropic[1], value)
    if (anthropic[2] === "remaining") return addRateLimitValue(remaining, anthropic[1], value)
    return addRateLimitValue(reset, anthropic[1], value)
  })

  if (
    retryAfter === undefined &&
    Object.keys(limit).length === 0 &&
    Object.keys(remaining).length === 0 &&
    Object.keys(reset).length === 0
  )
    return undefined

  return new HttpRateLimitDetails({
    retryAfterMs: retryAfter,
    limit: Object.keys(limit).length === 0 ? undefined : limit,
    remaining: Object.keys(remaining).length === 0 ? undefined : remaining,
    reset: Object.keys(reset).length === 0 ? undefined : reset,
  })
}

const requestDetails = (request: HttpClientRequest.HttpClientRequest, redactedNames: ReadonlyArray<string | RegExp>) =>
  new HttpRequestDetails({
    method: request.method,
    url: redactUrl(request.url),
    headers: redactHeaders(request.headers, redactedNames),
  })

const responseDetails = (
  response: HttpClientResponse.HttpClientResponse,
  redactedNames: ReadonlyArray<string | RegExp>,
) =>
  new HttpResponseDetails({
    status: response.status,
    headers: redactHeaders(response.headers, redactedNames),
  })

const secretValues = (request: HttpClientRequest.HttpClientRequest) => {
  const values = new Set<string>()
  const add = (value: string) => {
    if (value.length < 4) return
    values.add(value)
    values.add(encodeURIComponent(value))
    // A provider that echoes a secret back inside JSON sends it escaped, so the
    // literal pass has to look for that form as well as the raw one.
    values.add(JSON.stringify(value).slice(1, -1))
  }

  Object.entries(request.headers).forEach(([name, value]) => {
    if (!isSensitiveHeaderName(name)) return
    add(value)
    const bearer = /^Bearer\s+(.+)$/i.exec(value)?.[1]
    if (bearer) add(bearer)
  })

  if (!URL.canParse(request.url)) return values
  new URL(request.url).searchParams.forEach((value, key) => {
    if (isSensitiveQueryName(key)) add(value)
  })
  return values
}

// `secretValues` already refuses to track a value shorter than this, so a tail below
// it is not evidence of a secret. Stopping there also keeps an earlier pass's
// `<redacted>` from being eaten by a secret that happens to start with one of its
// own trailing characters.
const MIN_SECRET_TAIL = 4

// The literal pass needs a whole secret, so a secret the read cut in half survives it
// as a prefix sitting at the very end of the buffer — and redaction shrinking the text
// ahead of it can pull that prefix back under the reported budget. The longest suffix
// of the body that is a prefix of the secret is exactly the KMP automaton's state
// after consuming the body, which keeps the scan linear even when the secret and the
// body are both large.
const trailingSecretPrefixLength = (body: string, secret: string) => {
  const length = Math.min(secret.length - 1, body.length)
  if (length < MIN_SECRET_TAIL) return 0

  const prefix = secret.slice(0, length)
  const fallback = new Int32Array(length)
  let matched = 0
  for (let index = 1; index < prefix.length; index++) {
    while (matched > 0 && prefix[index] !== prefix[matched]) matched = fallback[matched - 1]
    if (prefix[index] === prefix[matched]) matched++
    fallback[index] = matched
  }

  matched = 0
  for (let index = 0; index < body.length; index++) {
    while (matched > 0 && body[index] !== prefix[matched]) matched = fallback[matched - 1]
    if (body[index] === prefix[matched]) matched++
  }
  return matched < MIN_SECRET_TAIL ? 0 : matched
}

// Two passes: structural (redact `"name": "value"` and `name=value` patterns
// for any field name that looks sensitive) plus literal (replace any actual
// secret values we sent in the request, in case the response echoes one back).
//
// The trailing repair measures every secret against the same text and cuts once at the
// longest match. Cutting per secret instead would let a short accidental overlap from
// one secret truncate the tail and hide the longer, real match belonging to another —
// leaving most of the cut secret in place.
//
// The repair is unconditional because the report is cut again at BODY_LIMIT, so a
// prefix can be exposed by a body that arrived whole and well inside the read budget.
// It covers a secret prefix that ends the redacted text — the read boundary, the
// transport cutting the body short, or a provider that truncated its own echo at the
// end. A self-truncated echo in the middle of a body is out of scope: catching that
// would mean redacting every long-enough factor of a secret. Over-redacting a tail that
// merely looks like the start of a secret is the safe direction.
const redactBody = (body: string, request: HttpClientRequest.HttpClientRequest) => {
  const structural = body
    .replace(REDACT_JSON_FIELD, `$1"${REDACTED}"`)
    .replace(REDACT_QUERY_FIELD, `$1${REDACTED}`)
    .replace(REDACT_UNTERMINATED_JSON_FIELD, `$1"${REDACTED}"`)
  const secrets = Array.from(secretValues(request))
  const redacted = secrets.reduce((text, secret) => text.split(secret).join(REDACTED), structural)

  const cut = secrets.reduce((longest, secret) => Math.max(longest, trailingSecretPrefixLength(redacted, secret)), 0)
  if (cut === 0) return redacted
  return `${redacted.slice(0, redacted.length - cut)}${REDACTED}`
}

// Provider error pages are unbounded and occasionally enormous, so only enough of the
// body to fill the diagnostic budget is buffered — each chunk is clipped to whatever
// budget is left, so one enormous chunk costs the budget rather than its own length,
// and the clipped pieces are appended rather than reconcatenated so the whole read
// stays O(n).
//
// Stopping early closes the stream's scope, which cancels the underlying reader, and a
// provider is free to hand back a cancel promise that never settles. Once the budget is
// full the buffer cannot grow again, so the read runs detached and the remainder
// cancellation is left to finish on its own instead of being awaited.
//
// Signalling that read uses `interruptUnsafe` rather than `Fiber.interrupt`, because
// `Fiber.interrupt` waits for the target to finish unwinding. Against a cancel that
// never settles it would never return, and forking it to avoid blocking would only
// trade a stuck caller for a second fiber stuck forever. `interruptUnsafe` is the same
// signal `Effect.runCallback`'s interruptor uses: synchronous, nothing to await, and no
// fiber created to deliver it.
//
// The signalled fiber unwinds into the stream's own reader-cancel finalizer, so against
// a cancel that never settles it stays parked there. Nothing returned from here refers
// to that fiber and the buffer is released once it has been copied, so the parked frame
// is reachable only through the provider's own pending cancel and holds nothing beyond
// the reader the provider is already holding: it is collected with the abandoned
// response rather than accumulating one live fiber per hostile error.
const errorBody = Effect.fnUntraced(function* (response: HttpClientResponse.HttpClientResponse) {
  const chunks: Uint8Array[] = []
  const full = yield* Deferred.make<void>()
  let size = 0

  const fiber = yield* response.stream.pipe(
    Stream.mapEffect((chunk) => {
      // Copied rather than viewed: a view over one enormous chunk would keep the
      // whole chunk alive, so the retained bytes would not be bounded by the budget.
      const slice = chunk.slice(0, BODY_READ_LIMIT - size)
      chunks.push(slice)
      size += slice.length
      if (size < BODY_READ_LIMIT) return Effect.succeed(false)
      return Deferred.succeed(full, undefined).pipe(Effect.as(true))
    }),
    Stream.takeUntil((done) => done),
    Stream.runDrain,
    Effect.forkDetach(),
  )
  // A body the transport cut short is indistinguishable from a complete one by size
  // alone, and it leaves behind exactly the same half-written values as a budget stop,
  // so the read reports whether it saw a natural end rather than how much it read. The
  // budget check stays as a tiebreak: when the last chunk fills the budget both branches
  // are ready at once, and the read still stopped early if the fiber wins the race.
  const complete = yield* Effect.raceFirst(
    Fiber.await(fiber).pipe(Effect.map(Exit.isSuccess)),
    Deferred.await(full).pipe(Effect.as(false)),
  ).pipe(Effect.ensuring(Effect.sync(() => fiber.interruptUnsafe())))

  if (size === 0) return undefined
  const bytes = new Uint8Array(size)
  chunks.reduce((offset, chunk) => {
    bytes.set(chunk, offset)
    return offset + chunk.length
  }, 0)
  // Released rather than left for the signalled read to drop: that read may still be
  // parked in a cancel the provider never settles, and it would otherwise keep the
  // whole buffer alive for as long as the provider keeps the stream.
  chunks.length = 0
  // A cut can land inside a multi-byte sequence, and decoding that tail would report a
  // real character as U+FFFD, so a clipped sequence is dropped instead.
  const end = !complete || size === BODY_READ_LIMIT ? completeScalarEnd(bytes) : size
  return { text: new TextDecoder().decode(bytes.subarray(0, end)), truncated: size > BODY_LIMIT }
})

// A UTF-8 lead byte encodes its own sequence length, and no sequence is longer than
// four bytes, so scanning back past the trailing continuation bytes is enough to tell
// a complete sequence from one the budget cut in half.
const completeScalarEnd = (bytes: Uint8Array) => {
  for (let index = bytes.length - 1; index >= 0 && index >= bytes.length - 4; index--) {
    const byte = bytes[index]
    if ((byte & 0b1100_0000) === 0b1000_0000) continue
    const length = byte < 0b1000_0000 ? 1 : byte >= 0b1111_0000 ? 4 : byte >= 0b1110_0000 ? 3 : 2
    return index + length <= bytes.length ? bytes.length : index
  }
  return bytes.length
}

// Truncation counts UTF-16 units, so a cut that lands between the two halves of an
// astral character would leave a lone surrogate behind; move it back one unit.
const sliceScalars = (value: string, limit: number) => {
  if (value.length <= limit) return value
  const lead = value.charCodeAt(limit - 1)
  return value.slice(0, lead >= 0xd800 && lead <= 0xdbff ? limit - 1 : limit)
}

const responseBody = (
  body: { readonly text: string; readonly truncated: boolean } | undefined,
  request: HttpClientRequest.HttpClientRequest,
) => {
  if (body === undefined) return {}
  const redacted = redactBody(body.text, request)
  if (!body.truncated && redacted.length <= BODY_LIMIT) return { body: redacted }
  return { body: sliceScalars(redacted, BODY_LIMIT), bodyTruncated: true }
}

// HTML/gateway error pages are useless (and noisy) in TUI retry notices (#35640).
const isHtmlErrorBody = (body: string) => /^\s*<(!doctype\s+html|html[\s>])/i.test(body)

const providerMessage = (status: number, body: { readonly body?: string }) => {
  const text = body.body
  if (!text || text.length > 500 || isHtmlErrorBody(text)) {
    return `Provider request failed with HTTP ${status}`
  }
  return `Provider request failed with HTTP ${status}: ${text}`
}

const responseHttp = (input: {
  readonly request: HttpClientRequest.HttpClientRequest
  readonly response: HttpClientResponse.HttpClientResponse
  readonly redactedNames: ReadonlyArray<string | RegExp>
  readonly body: ReturnType<typeof responseBody>
  readonly requestId?: string | undefined
  readonly rateLimit?: HttpRateLimitDetails | undefined
}) =>
  new HttpContext({
    request: requestDetails(input.request, input.redactedNames),
    response: responseDetails(input.response, input.redactedNames),
    ...input.body,
    requestId: input.requestId,
    rateLimit: input.rateLimit,
  })

const statusReason = (input: {
  readonly status: number
  readonly message: string
  readonly retryAfterMs?: number | undefined
  readonly rateLimit?: HttpRateLimitDetails | undefined
  readonly http: HttpContext
}) => {
  const body = input.http.body ?? ""
  if (/content[-_\s]?policy|content_filter|safety/i.test(body)) {
    return new ContentPolicyReason({ message: input.message, http: input.http })
  }
  if (input.status === 401) {
    return new AuthenticationReason({ message: input.message, kind: "invalid", http: input.http })
  }
  if (input.status === 403) {
    return new AuthenticationReason({ message: input.message, kind: "insufficient-permissions", http: input.http })
  }
  if (input.status === 429) {
    if (/insufficient[-_\s]?quota|quota[-_\s]?exceeded/i.test(body)) {
      return new QuotaExceededReason({ message: input.message, http: input.http })
    }
    return new RateLimitReason({
      message: input.message,
      retryAfterMs: input.retryAfterMs,
      rateLimit: input.rateLimit,
      http: input.http,
    })
  }
  if (
    input.status === 400 ||
    input.status === 404 ||
    input.status === 409 ||
    input.status === 413 ||
    input.status === 422
  ) {
    return new InvalidRequestReason({
      message: input.message,
      classification: isContextOverflow(body) ? "context-overflow" : undefined,
      http: input.http,
    })
  }
  if (input.status >= 500 || retryableStatus(input.status)) {
    return new ProviderInternalReason({
      message: input.message,
      status: input.status,
      retryAfterMs: input.retryAfterMs,
      http: input.http,
    })
  }
  return new UnknownProviderReason({ message: input.message, status: input.status, http: input.http })
}

const statusError =
  (request: HttpClientRequest.HttpClientRequest, redactedNames: ReadonlyArray<string | RegExp>) =>
  (response: HttpClientResponse.HttpClientResponse) =>
    Effect.gen(function* () {
      if (response.status < 400) return response
      const body = yield* errorBody(response)
      const headers = normalizedHeaders(response.headers)
      const retryAfter = retryAfterMs(headers)
      const rateLimit = rateLimitDetails(headers, retryAfter)
      const details = responseBody(body, request)
      return yield* new LLMError({
        module: "RequestExecutor",
        method: "execute",
        reason: statusReason({
          status: response.status,
          message: providerMessage(response.status, details),
          retryAfterMs: retryAfter,
          rateLimit,
          http: responseHttp({
            request,
            response,
            redactedNames,
            body: details,
            requestId: requestId(headers),
            rateLimit,
          }),
        }),
      })
    })

const toHttpError = (redactedNames: ReadonlyArray<string | RegExp>) => (error: unknown) => {
  const transportError = (input: {
    readonly message: string
    readonly kind?: string | undefined
    readonly request?: HttpClientRequest.HttpClientRequest | undefined
  }) =>
    new LLMError({
      module: "RequestExecutor",
      method: "execute",
      reason: new TransportReason({
        message: input.message,
        kind: input.kind,
        url: input.request ? redactUrl(input.request.url) : undefined,
        http: input.request ? new HttpContext({ request: requestDetails(input.request, redactedNames) }) : undefined,
      }),
    })

  if (Cause.isTimeoutError(error)) {
    return transportError({ message: error.message, kind: "Timeout" })
  }
  if (!HttpClientError.isHttpClientError(error)) {
    return transportError({ message: "HTTP transport failed" })
  }
  const request = "request" in error ? error.request : undefined
  if (error.reason._tag === "TransportError") {
    return transportError({
      message: error.reason.description ?? "HTTP transport failed",
      kind: error.reason._tag,
      request,
    })
  }
  return transportError({
    message: `HTTP transport failed: ${error.reason._tag}`,
    kind: error.reason._tag,
    request,
  })
}

const retryDelay = (error: LLMError, attempt: number) => {
  if (error.retryAfterMs !== undefined) return Effect.succeed(Math.min(error.retryAfterMs, MAX_DELAY_MS))
  return Random.nextBetween(
    Math.min(BASE_DELAY_MS * 2 ** attempt * 0.8, MAX_DELAY_MS),
    Math.min(BASE_DELAY_MS * 2 ** attempt * 1.2, MAX_DELAY_MS),
  ).pipe(Effect.map((delay) => Math.round(delay)))
}

const retryStatusFailures = <A, R>(
  effect: Effect.Effect<A, LLMError, R>,
  retries = MAX_RETRIES,
  attempt = 0,
): Effect.Effect<A, LLMError, R> =>
  Effect.catchTag(effect, "LLM.Error", (error): Effect.Effect<A, LLMError, R> => {
    if (!error.retryable || retries <= 0) return Effect.fail(error)
    return retryDelay(error, attempt).pipe(
      Effect.flatMap((delay) => Effect.sleep(delay)),
      Effect.flatMap(() => retryStatusFailures(effect, retries - 1, attempt + 1)),
    )
  })

export const layer: Layer.Layer<Service, never, HttpClient.HttpClient> = Layer.effect(
  Service,
  Effect.gen(function* () {
    const http = yield* HttpClient.HttpClient
    const executeOnce = (request: HttpClientRequest.HttpClientRequest) =>
      Effect.gen(function* () {
        const redactedNames = yield* Headers.CurrentRedactedNames
        return yield* http
          .execute(request)
          .pipe(Effect.mapError(toHttpError(redactedNames)), Effect.flatMap(statusError(request, redactedNames)))
      })
    return Service.of({
      execute: (request) => retryStatusFailures(executeOnce(request)),
    })
  }),
)

export const fetchLayer = layer.pipe(Layer.provide(FetchHttpClient.layer))

export * as RequestExecutor from "./executor"
