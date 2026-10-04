import { text } from "node:stream/consumers"
import type { Writable } from "node:stream"

// Only used immediately before exit: Bun's empty-write callback can run while
// earlier stdout writes are still pending, so finish the stream to drain them.
export function flushOutput(stream: Writable) {
  if (stream.destroyed || stream.writableFinished) return Promise.resolve()
  return new Promise<void>((resolve) => {
    const done = () => {
      stream.off("error", done)
      stream.off("finish", done)
      stream.off("close", done)
      resolve()
    }
    stream.once("error", done)
    stream.once("finish", done)
    stream.once("close", done)
    stream.end()
  })
}

export function readStdin() {
  return text(process.stdin)
}
