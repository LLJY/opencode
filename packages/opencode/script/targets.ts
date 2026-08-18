export type Target = {
  os: string
  arch: "arm64" | "x64"
  abi?: "musl"
  avx2?: false
}

export const allTargets: Target[] = [
  { os: "linux", arch: "arm64" },
  { os: "linux", arch: "x64" },
  { os: "linux", arch: "x64", avx2: false },
  { os: "linux", arch: "arm64", abi: "musl" },
  { os: "linux", arch: "x64", abi: "musl" },
  { os: "linux", arch: "x64", abi: "musl", avx2: false },
  { os: "darwin", arch: "arm64" },
  { os: "darwin", arch: "x64" },
  { os: "darwin", arch: "x64", avx2: false },
  { os: "win32", arch: "arm64" },
  { os: "win32", arch: "x64" },
  { os: "win32", arch: "x64", avx2: false },
]

export const targetName = (name: string, target: Target) =>
  [
    name,
    target.os === "win32" ? "windows" : target.os,
    target.arch,
    target.avx2 === false ? "baseline" : undefined,
    target.abi,
  ]
    .filter(Boolean)
    .join("-")

export function selectTargets(input: { name: string; args: string[]; platform: string; arch: string }) {
  const targetArgs = input.args.filter((arg) => arg.startsWith("--target"))
  if (targetArgs.length > 1) throw new Error("Build target may be specified only once")
  if (targetArgs[0] && !targetArgs[0].startsWith("--target="))
    throw new Error(`Invalid build target argument: ${targetArgs[0]}`)
  const target = targetArgs[0]?.slice("--target=".length)
  if (targetArgs[0] && !target) throw new Error("Build target must not be empty")

  const selected = target
    ? allTargets.filter((item) => targetName(input.name, item).slice(`${input.name}-`.length) === target)
    : input.args.includes("--single")
      ? allTargets.filter(
          (item) =>
            item.os === input.platform &&
            item.arch === input.arch &&
            item.abi === undefined &&
            (item.avx2 !== false || input.args.includes("--baseline")),
        )
      : allTargets
  if (selected.length === 0) throw new Error(`Unknown or unavailable build target: ${target ?? "native"}`)
  return selected
}
