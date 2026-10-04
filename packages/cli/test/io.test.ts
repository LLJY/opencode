import { expect, test } from "bun:test"
import { Writable } from "node:stream"
import { flushOutput } from "../src/util/io"

test("flush cleans up its failure listeners on success, EPIPE and write failures", async () => {
  for (const code of [undefined, "EPIPE", "EIO"]) {
    const stream = new Writable({
      write(_chunk, _encoding, callback) {
        setImmediate(() => callback(code ? Object.assign(new Error(code), { code }) : undefined))
      },
    })
    stream.write("pending output")
    await flushOutput(stream)
    await new Promise<void>((resolve) => setImmediate(resolve))
    expect(stream.listenerCount("error")).toBe(0)
    expect(stream.listenerCount("finish")).toBe(0)
    expect(stream.listenerCount("close")).toBe(0)
  }
})
