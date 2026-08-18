import { describe, expect, it } from "bun:test"
import path from "node:path"

const promptDir = path.join(import.meta.dir, "../../src/session/prompt")

describe("system prompt files", () => {
  it("do not reference the multi_tool_use.parallel pseudo-tool", async () => {
    const files = await Array.fromAsync(new Bun.Glob("*.txt").scan(promptDir))
    expect(files.length).toBeGreaterThan(0)
    const prompts = await Promise.all(
      files.map(async (file) => [file, await Bun.file(path.join(promptDir, file)).text()] as const),
    )
    prompts.forEach(([file, content]) =>
      expect(content, `${file} references multi_tool_use.parallel`).not.toContain("multi_tool_use.parallel"),
    )
  })
})
