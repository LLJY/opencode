const DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
const DAY_NAME = "Mon|Tue|Wed|Thu|Fri|Sat|Sun"
const DAY_NAME_L = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
const MONTH = MONTHS.join("|")
const TIME = String.raw`(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2})`

// The three HTTP-date forms every recipient must accept (RFC 9110 5.6.7). They are
// case sensitive and fixed width. `Date.parse` cannot stand in for these patterns:
// it accepts arbitrary prose such as "January 1, 2099", which turns a malformed
// Retry-After into a multi-week wait, and it reads asctime stamps as local time
// even though the grammar defines them as UTC.
const IMF_FIXDATE = new RegExp(
  String.raw`^(?<weekday>${DAY_NAME}), (?<day>\d{2}) (?<month>${MONTH}) (?<year>\d{4}) ${TIME} GMT$`,
)
const RFC850_DATE = new RegExp(
  String.raw`^(?<weekday>${DAY_NAME_L}), (?<day>\d{2})-(?<month>${MONTH})-(?<year>\d{2}) ${TIME} GMT$`,
)
const ASCTIME_DATE = new RegExp(
  String.raw`^(?<weekday>${DAY_NAME}) (?<month>${MONTH}) (?<day>[ \d]\d) ${TIME} (?<year>\d{4})$`,
)

// Two-digit years only ever expand into 1900-2099, and a timestamp older than the
// HTTP/1.0 era is a parser accident rather than a real Retry-After.
const MIN_YEAR = 1900

/**
 * Parses an RFC 9110 HTTP-date to epoch milliseconds. Returns undefined for any other input.
 * `value` must already be trimmed; `now` is the reference instant for the rfc850 two-digit year rule.
 *
 * A leap second (`60`) is accepted and normalized forward into the following minute,
 * so `23:59:60` becomes the next day at `00:00:00`.
 */
export function parse(value: string, now: number) {
  const groups = (IMF_FIXDATE.exec(value) ?? RFC850_DATE.exec(value) ?? ASCTIME_DATE.exec(value))?.groups
  if (!groups) return undefined

  const day = Number(groups.day)
  const hour = Number(groups.hour)
  const minute = Number(groups.minute)
  const second = Number(groups.second)
  if (day < 1 || day > 31 || hour > 23 || minute > 59 || second > 60) return undefined

  const year =
    groups.year.length === 4
      ? Number(groups.year)
      : expandTwoDigitYear({
          year: Number(groups.year),
          month: MONTHS.indexOf(groups.month),
          day,
          hour,
          minute,
          second,
          now,
        })
  if (year < MIN_YEAR) return undefined

  // Resolve the calendar day on its own so the time of day cannot mask an invalid
  // date, and so a leap second stays a pure offset rather than a rolled field.
  const midnight = Date.UTC(year, MONTHS.indexOf(groups.month), day)
  const at = new Date(midnight)
  // Date.UTC rolls impossible days ("31 Nov" -> 01 Dec), so compare the round trip.
  if (at.getUTCFullYear() !== year || at.getUTCMonth() !== MONTHS.indexOf(groups.month) || at.getUTCDate() !== day) {
    return undefined
  }
  // The weekday is redundant information the sender must still get right; a mismatch
  // means the value was hand-built or corrupted and cannot be trusted as a deadline.
  if (DAY_NAMES[at.getUTCDay()] !== groups.weekday.slice(0, 3)) return undefined

  return midnight + (hour * 3_600 + minute * 60 + second) * 1_000
}

// RFC 9110 5.6.7: an rfc850-date two-digit year that looks more than 50 years
// ahead means the most recent past year ending in the same two digits.
function expandTwoDigitYear(input: {
  readonly year: number
  readonly month: number
  readonly day: number
  readonly hour: number
  readonly minute: number
  readonly second: number
  readonly now: number
}) {
  const current = new Date(input.now)
  const candidate = Math.floor(current.getUTCFullYear() / 100) * 100 + input.year
  const threshold = new Date(input.now)
  threshold.setUTCFullYear(current.getUTCFullYear() + 50)
  const timestamp = Date.UTC(candidate, input.month, input.day, input.hour, input.minute, input.second)
  return timestamp > threshold.getTime() ? candidate - 100 : candidate
}

export * as HttpDate from "./http-date"
