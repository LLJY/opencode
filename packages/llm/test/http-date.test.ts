import { describe, expect, test } from "bun:test"
import { parseHttpDate } from "../src/utils/http-date"

// Mirrors packages/opencode/test/util/http-date.test.ts; keep both suites in sync.
const now = Date.UTC(2026, 0, 2, 3, 4, 5)
const reference = Date.UTC(1994, 10, 6, 8, 49, 37)

describe("parseHttpDate", () => {
  test("parses the three HTTP-date forms as UTC", () => {
    expect(parseHttpDate("Sun, 06 Nov 1994 08:49:37 GMT", now)).toBe(reference)
    expect(parseHttpDate("Sunday, 06-Nov-94 08:49:37 GMT", now)).toBe(reference)
    // asctime carries no zone; the grammar defines it as UTC, unlike Date.parse.
    expect(parseHttpDate("Sun Nov  6 08:49:37 1994", now)).toBe(reference)
    expect(parseHttpDate("Wed Nov 16 08:49:37 1994", now)).toBe(Date.UTC(1994, 10, 16, 8, 49, 37))
  })

  test("resolves rfc850 two-digit years against the 50-year rule", () => {
    expect(parseHttpDate("Saturday, 06-Nov-27 08:49:37 GMT", now)).toBe(Date.UTC(2027, 10, 6, 8, 49, 37))
    expect(parseHttpDate("Thursday, 02-Jan-76 03:04:05 GMT", now)).toBe(Date.UTC(2076, 0, 2, 3, 4, 5))
    // One second beyond the complete 50-year timestamp boundary rolls back a century.
    expect(parseHttpDate("Friday, 02-Jan-76 03:04:06 GMT", now)).toBe(Date.UTC(1976, 0, 2, 3, 4, 6))
    expect(parseHttpDate("Saturday, 06-Nov-76 08:49:37 GMT", now)).toBe(Date.UTC(1976, 10, 6, 8, 49, 37))
    expect(parseHttpDate("Sunday, 06-Nov-77 08:49:37 GMT", now)).toBe(Date.UTC(1977, 10, 6, 8, 49, 37))
  })

  test("accepts a real leap day", () => {
    expect(parseHttpDate("Thu, 29 Feb 2024 12:00:00 GMT", now)).toBe(Date.UTC(2024, 1, 29, 12, 0, 0))
  })

  test("normalizes an explicit leap second into the next minute", () => {
    expect(parseHttpDate("Sun, 06 Nov 1994 23:59:60 GMT", now)).toBe(Date.UTC(1994, 10, 7, 0, 0, 0))
    expect(parseHttpDate("Sun, 06 Nov 1994 08:49:60 GMT", now)).toBe(Date.UTC(1994, 10, 6, 8, 50, 0))
  })

  test.each([
    // Weekday matches the date Date.UTC would roll to, so only a calendar check rejects these.
    "Fri, 30 Feb 2024 12:00:00 GMT",
    "Thu, 31 Nov 1994 08:49:37 GMT",
    "Wed, 31 Apr 2026 08:49:37 GMT",
    // 2023 was not a leap year.
    "Wed, 29 Feb 2023 12:00:00 GMT",
  ])("rejects impossible calendar date %p", (value) => {
    expect(parseHttpDate(value, now)).toBeUndefined()
  })

  test.each(["Mon, 06 Nov 1994 08:49:37 GMT", "Monday, 06-Nov-94 08:49:37 GMT", "Mon Nov  6 08:49:37 1994"])(
    "rejects weekday that disagrees with the date %p",
    (value) => {
      expect(parseHttpDate(value, now)).toBeUndefined()
    },
  )

  test.each([
    // Date.UTC maps 0-99 onto 1900-1999, so a four-digit year below 1900 is a parser accident.
    "Sun, 06 Nov 0094 08:49:37 GMT",
    "Mon, 01 Jan 0001 00:00:00 GMT",
    "Sun, 31 Dec 1899 23:59:59 GMT",
  ])("rejects four-digit years below 1900 %p", (value) => {
    expect(parseHttpDate(value, now)).toBeUndefined()
  })

  test.each([
    "January 1, 2099",
    "Jan 1, 2099",
    "1 January 2099",
    "2099-01-01T00:00:00Z",
    "Tomorrow",
    "next friday",
    "sun, 06 nov 1994 08:49:37 gmt",
    "Sun, 06 Nov 1994 08:49:37 UTC",
    "Sun, 6 Nov 1994 08:49:37 GMT",
    "Sun, 06 Nov 1994 25:49:37 GMT",
    "Sun, 06 Nov 1994 08:60:37 GMT",
    "Sun, 06 Nov 1994 08:49:61 GMT",
    "Sun, 32 Nov 1994 08:49:37 GMT",
    "Sun, 00 Nov 1994 08:49:37 GMT",
    // Callers are responsible for trimming; surrounding whitespace is not part of the grammar.
    " Sun, 06 Nov 1994 08:49:37 GMT",
    "Sun, 06 Nov 1994 08:49:37 GMT ",
    "",
  ])("rejects malformed HTTP-date %p", (value) => {
    expect(parseHttpDate(value, now)).toBeUndefined()
  })
})
