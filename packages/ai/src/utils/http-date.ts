const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
const DAY = DAYS.join("|")
const LONG_DAY = "Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday"
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
const MONTH = MONTHS.join("|")
const TIME = String.raw`(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2})`
const IMF = new RegExp(String.raw`^(?<weekday>${DAY}), (?<day>\d{2}) (?<month>${MONTH}) (?<year>\d{4}) ${TIME} GMT$`)
const RFC850 = new RegExp(
  String.raw`^(?<weekday>${LONG_DAY}), (?<day>\d{2})-(?<month>${MONTH})-(?<year>\d{2}) ${TIME} GMT$`,
)
const ASCTIME = new RegExp(String.raw`^(?<weekday>${DAY}) (?<month>${MONTH}) (?<day>[ \d]\d) ${TIME} (?<year>\d{4})$`)

/** RFC 9110 HTTP-date, including obsolete recipient forms, in UTC. Never parse arbitrary date prose. */
export const parseHttpDate = (value: string, now: number): number | undefined => {
  const fields = (IMF.exec(value) ?? RFC850.exec(value) ?? ASCTIME.exec(value))?.groups
  if (!fields) return undefined
  const day = Number(fields.day)
  const month = MONTHS.indexOf(fields.month!)
  const hour = Number(fields.hour)
  const minute = Number(fields.minute)
  const second = Number(fields.second)
  if (day < 1 || day > 31 || hour > 23 || minute > 59 || second > 60) return undefined
  const candidate =
    fields.year!.length === 4
      ? Number(fields.year)
      : Math.floor(new Date(now).getUTCFullYear() / 100) * 100 + Number(fields.year)
  const threshold = new Date(now)
  threshold.setUTCFullYear(threshold.getUTCFullYear() + 50)
  const year =
    fields.year!.length === 2 && Date.UTC(candidate, month, day, hour, minute, second) > threshold.getTime()
      ? candidate - 100
      : candidate
  if (year < 1900) return undefined
  const midnight = Date.UTC(year, month, day)
  const at = new Date(midnight)
  if (at.getUTCFullYear() !== year || at.getUTCMonth() !== month || at.getUTCDate() !== day) return undefined
  if (DAYS[at.getUTCDay()] !== fields.weekday!.slice(0, 3)) return undefined
  // Leap seconds are offsets, not rolled calendar fields.
  return midnight + (hour * 3600 + minute * 60 + second) * 1000
}
