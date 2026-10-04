import { expect, test } from "bun:test"
import path from "node:path"
import { tmpdir } from "./fixture/tmpdir"
import { isolatedEnv } from "./fixture/environment"

test("build scripts validate flags and targets before touching output or building", async () => {
  for (const script of ["build.ts", "build-node.ts"]) {
    for (const input of [
      {
        flags: ["--target=not-a-target"],
        error: script === "build.ts" ? "Unknown build target" : "Unknown Node target",
      },
      { flags: ["--mistyped", "--target=not-a-target"], error: "Unknown build flag: --mistyped" },
      { flags: ["--target=not-a-target", "--target=also-invalid"], error: "Duplicate build flag: --target" },
      { flags: ["--target="], error: "--target requires a value" },
    ]) {
      await using root = await tmpdir()
      const sentinel = path.join(root.path, "keep.txt")
      await Bun.write(sentinel, "keep existing output")
      const child = Bun.spawn(
        [
          process.execPath,
          "run",
          path.join(import.meta.dir, "../script", script),
          `--outdir=${root.path}`,
          "--skip-install",
          ...input.flags,
        ],
        {
          cwd: path.join(import.meta.dir, ".."),
          env: isolatedEnv(root.path, { OPENCODE_VERSION: "2.0.21", OPENCODE_CHANNEL: "local" }),
          stdout: "pipe",
          stderr: "pipe",
        },
      )
      const [stdout, stderr, exitCode] = await Promise.all([
        new Response(child.stdout).text(),
        new Response(child.stderr).text(),
        child.exited,
      ])
      expect(exitCode).not.toBe(0)
      expect(stderr).toContain(input.error)
      expect(stdout).not.toContain("building ")
      expect(await Bun.file(sentinel).exists()).toBe(true)
      expect(await Bun.file(sentinel).text()).toBe("keep existing output")
    }
  }
}, 30_000)
