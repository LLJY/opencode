import { describe, expect, test } from "bun:test"
import { allTargets, selectTargets, targetName } from "../../script/targets"

const select = (args: string[], platform = "linux", arch = "x64") =>
  selectTargets({ name: "opencode", args, platform, arch }).map((target) => targetName("opencode", target))

describe("build targets", () => {
  test("names all supported target variants", () => {
    expect(allTargets.map((target) => targetName("opencode", target))).toEqual([
      "opencode-linux-arm64",
      "opencode-linux-x64",
      "opencode-linux-x64-baseline",
      "opencode-linux-arm64-musl",
      "opencode-linux-x64-musl",
      "opencode-linux-x64-baseline-musl",
      "opencode-darwin-arm64",
      "opencode-darwin-x64",
      "opencode-darwin-x64-baseline",
      "opencode-windows-arm64",
      "opencode-windows-x64",
      "opencode-windows-x64-baseline",
    ])
  })

  test.each([
    "linux-arm64",
    "linux-x64",
    "linux-x64-baseline",
    "linux-arm64-musl",
    "linux-x64-musl",
    "linux-x64-baseline-musl",
    "darwin-arm64",
    "darwin-x64",
    "darwin-x64-baseline",
    "windows-arm64",
    "windows-x64",
    "windows-x64-baseline",
  ])("selects explicit target %s", (target) => {
    expect(select([`--target=${target}`, "--single", "--baseline"])).toEqual([`opencode-${target}`])
  })

  test("selects one native target unless baseline is requested", () => {
    expect(select(["--single"])).toEqual(["opencode-linux-x64"])
    expect(select(["--single", "--baseline"])).toEqual(["opencode-linux-x64", "opencode-linux-x64-baseline"])
  })

  // ARM64 has no baseline variant to add, so asking for one must still resolve to the
  // single native target rather than to nothing.
  test("selects the native ARM64 target with or without baseline", () => {
    expect(select(["--single"], "linux", "arm64")).toEqual(["opencode-linux-arm64"])
    expect(select(["--single", "--baseline"], "linux", "arm64")).toEqual(["opencode-linux-arm64"])
    expect(select(["--single", "--baseline"], "darwin", "arm64")).toEqual(["opencode-darwin-arm64"])
    expect(select(["--single", "--baseline"], "win32", "arm64")).toEqual(["opencode-windows-arm64"])
  })

  test("rejects invalid target arguments", () => {
    expect(() => select(["--target=linux-x64", "--target=linux-arm64"])).toThrow(
      "Build target may be specified only once",
    )
    expect(() => select(["--target"])).toThrow("Invalid build target argument: --target")
    expect(() => select(["--target="])).toThrow("Build target must not be empty")
    expect(() => select(["--target=linux-x86"])).toThrow("Unknown or unavailable build target: linux-x86")
    expect(() => select(["--single"], "freebsd")).toThrow("Unknown or unavailable build target: native")
  })
})
